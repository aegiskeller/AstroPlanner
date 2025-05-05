import requests
import ephem
import json
import datetime
import PES_secrets
import pandas as pd
from zoneinfo import ZoneInfo
import math

def check_auth():
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/telescope",auth=(PES_secrets.apikey,"api_token"), timeout=5)
    # check if the requests raises an exception
    if entry.status_code != 200:
        print(f"Error: {entry.status_code}")
        print("Please check your API key and try again.")
        return None
    # check if the response is empty
    if entry.json() == {}:
        print("Error: Site is not defined for this API key")
        print("Please check your configuration and try again.")
        return None
    return(entry.json())

def get_sun_rise_set(datestr):
    # lets now consider the night in question
    # Make an observer
    planobs = ephem.Observer()

    # Set the date and time 
    planobs.date = datestr

    # Location 
    planobs.lon = str(PES_secrets.obslon)
    planobs.lat = str(PES_secrets.obslat)

    # Elevation 
    planobs.elevation = PES_secrets.obsalt
    # To get U.S. Naval Astronomical Almanac values, use these settings
    planobs.pressure = 0
    planobs.horizon = '-0:34'

    # Calculate sunrise, solar noon, and sunset
    sunrise = planobs.next_rising(ephem.Sun(), use_center=True)
    sunset = planobs.next_setting(ephem.Sun(), use_center=True)

    # Relocate the horizon to get twilight times
    planobs.horizon = str(PES_secrets.obshorizon)  # -6=civil twilight, -12=nautical, -18=astronomical
    beg_twilight = planobs.next_rising(ephem.Sun(), use_center=True)
    end_twilight = planobs.next_setting(ephem.Sun(), use_center=True)
    # check the times are not the same
    if beg_twilight == end_twilight:
        print("Error: The times for twilight are the same, check the date")
        return None
    zone = ZoneInfo('GMT')
    return(ephem.to_timezone(sunset, zone).timestamp(), 
           ephem.to_timezone(sunrise, zone).timestamp())

def get_targets():
    obs_sections = PES_secrets.obs_sections
    minalt = PES_secrets.minalt # degrees above horizon
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/targets",\
                     auth=(PES_secrets.apikey,"api_token"),params={'observable':['True'],'obs_section':[obs_sections], 'targetaltitude':[str(minalt)], 'orderby':['ra']}, timeout=5)
    
    temp_json = entry.json()
    df = pd.DataFrame.from_dict(temp_json['targets'])
    # convert last_data_point from a seconds timestamp  to datetime object and format as YYYY-MM-DD
    df['last_data_point'] = pd.to_datetime(df['last_data_point'], unit='s').dt.strftime('%Y-%m-%d')
    return df

def parse_ephemeris(ref):
    #print(ref)
    # ref looks like [[Ephemeris info https://www.aavso.org/vsx/index.php?view=detail.ephemeris&nolayout=1&oid=469680]]
    if not ref:
        return None
    ref = ref.split(' ')
    # get the url from the ref
    url = ref[2]
    # remove the trailing ']]'
    url = url[:-2]
    # get the ephemeris from the url
    try:
        entry = requests.get(url, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"Info: {e}")
        return None
    # parse the ephemeris to a pandas dataframe
    #<td align="left">2460515.239</td><td align="left">23 Jul 2024 17:44</td></td></tr>
    # <tr><td class="windowtitle" align="right" valign="middle" height="2" colspan="2"/></tr>
    # <tr>
    ephemeris = []
    for line in entry.iter_lines():
        # if line contains the ephemeris info
        if b'<td align="left">' in line:
            # parse the line to get the date and time
            line = line.decode('utf-8')
            line = line.split('<td align="left">')
            # get the date and time
            dtstr = line[2].split('</td>')[0]
            # convert the date and time to a datetime object
            dt = datetime.datetime.strptime(dtstr, '%d %b %Y %H:%M')
            # store the date and time in a list
            ephemeris.append(dt)
    return ephemeris

def event_tonight(targetdf):
    # for each target check if there is an event tonight
    # if there is an event then we add a column to the targetdf
    # with the event time
    for index, row in targetdf.iterrows():
        # check if the ephemerise is empty
        if row['ephemeris'] is None:
            # there is no event tonight
            # add a column to the targetdf with the value None
            targetdf.at[index, 'event'] = None
            continue
        # if the ephemeris is not empty then we check if there is an event tonight
        # get the ephemeris
        ephemeris = row['ephemeris']
        # create a new column in the targetdf
        targetdf.at[index, 'event'] = None
        # does the ephemeris contain a datetime that is between the sunset and sunrise
        # get the sunset and sunrise times
        sunset, sunrise = get_sun_rise_set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # check if the ephemeris is between the sunset and sunrise
        for event in ephemeris:
            # check if the event is between the sunset and sunrise
            if event.timestamp() > sunset and event.timestamp() < sunrise:
                # add the event to the targetdf
                targetdf.at[index, 'event'] = event
                # print the event
                print(f"Event: {event} for {row['star_name']}")
                break
            # also locate the event that occurs next
            # check if the event is after the sunrise
            if event.timestamp() > sunrise:
                # add the event to the targetdf - add the local time pof the event
                targetdf.at[index, 'next_event'] = event.timestamp().tz_convert(PES_secrets.timezone)    
                break
    return targetdf

def get_target_airmass(target):
    """
    determine the airmass for a target over the course of the night
    input is a row of the target dataframe as recieved from the get_targets function
    """
    ra = target['ra']
    dec = target['dec']
    # get the sunset and sunrise times
    sunset, sunrise = get_sun_rise_set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # create a list of times from sunset to sunrise - sampled every 1h
    times = []
    for i in range(0, 24):
        # create a time object
        time = datetime.datetime.fromtimestamp(sunset) + datetime.timedelta(hours=i)
        # check if the time is before sunrise
        if time.timestamp() < sunrise:
            times.append(time.timestamp())
    airmass = []
    for time in times:
        # create an observer
        planobs = ephem.Observer()
        # set the date and time 
        planobs.date = datetime.datetime.fromtimestamp(time)
        # Location 
        planobs.lon = str(PES_secrets.obslon)
        planobs.lat = str(PES_secrets.obslat)
        # Elevation 
        planobs.elevation = PES_secrets.obsalt
        # To get U.S. Naval Astronomical Almanac values, use these settings
        planobs.pressure = 0
        planobs.horizon = '-0:34'
        # create a star object
        star = ephem.FixedBody()
        star._epoch = ephem.J2000
        star._ra = ra/180.0*ephem.pi
        star._dec = dec/180.0*ephem.pi
        star.compute(planobs)
        # calculate the airmass
        airmass.append(1/math.cos(math.radians(90.0 - star.alt*180.0/ephem.pi)))
        #print(planobs.date, ra, dec)
    # return the airmass
    return times, airmass

def determine_moon_phase(date):
    """
    determine the moon phase for a given date
    input is a datetime object
    """
    # get the moon phase
    moon = ephem.Moon()
    # create an observer
    planobs = ephem.Observer()
    # set the date and time 
    planobs.date = date
    # Location 
    planobs.lon = str(PES_secrets.obslon)
    planobs.lat = str(PES_secrets.obslat)
    # Elevation 
    planobs.elevation = PES_secrets.obsalt
    # To get U.S. Naval Astronomical Almanac values, use these settings
    planobs.pressure = 0
    planobs.horizon = '-0:34'
    # compute the moon phase
    moon.compute(planobs)
    return moon.phase