import requests
import ephem
import json
import datetime
import PES_secrets
import pandas as pd
from zoneinfo import ZoneInfo

def check_auth():
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/telescope",auth=(PES_secrets.apikey,"api_token"), timeout=5)
    return(entry.json())

def get_horizon(datestr):
    # lets now consider the night in question
    # Make an observer
    wombat = ephem.Observer()

    # Set the date and time 
    wombat.date = datestr

    # Location 
    wombat.lon = str(148.12899)
    wombat.lat = str(-35.282)

    # Elevation 
    wombat.elev = 500

    # To get U.S. Naval Astronomical Almanac values, use these settings
    wombat.pressure = 0
    wombat.horizon = '-0:34'

    # Calculate sunrise, solar noon, and sunset
    sunrise = wombat.next_rising(ephem.Sun(), use_center=True)
    sunset = wombat.next_setting(ephem.Sun(), use_center=True)

    # Relocate the horizon to get twilight times
    wombat.horizon = '-12'  # -6=civil twilight, -12=nautical, -18=astronomical
    beg_twilight = wombat.next_rising(ephem.Sun(), use_center=True)
    end_twilight = wombat.next_setting(ephem.Sun(), use_center=True)
    # check the times are not the same
    if beg_twilight == end_twilight:
        print("Error: The times for twilight are the same, check the date")
        return None

    #print(f"Sunset: {sunset}")
    #print(f"Sunrise: {sunrise}")

    zone = ZoneInfo('GMT')
    return(ephem.to_timezone(sunset, zone).timestamp(), 
           ephem.to_timezone(sunrise, zone).timestamp())

def get_targets():
    obs_sections = ['eb'] #ac,ep,cv,eb,spp,lpv,yso,het,misc,all.
    minalt = 30 # degrees above horizon
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/targets",\
                     auth=(PES_secrets.apikey,"api_token"),params={'observable':['True'],'obs_section':[obs_sections], 'targetaltitude':[str(minalt)], 'orderby':['ra']}, timeout=5)
    return(entry.json())

def parse_targets():
    # the json looks like this
    # {'star_name': 'V1723 Sco', 'ra': 261.57529, 'dec': -38.16008, 'constellation': 'Sco', 'var_type': 'NA', 'min_mag': None, 'min_mag_band': 'V', 'max_mag': 6.8, 'max_mag_band': 'V', 'period': None, 'obs_cadence': 3.0, 'obs_mode': 'All', 'obs_section': ['Alerts / Campaigns', 'Cataclysmic Variables', 'Eclipsing Variables', 'Short Period Pulsators', 'Long Period Variables', 'Young Stellar Objects'], 'filter': 'R', 'other_info': 'N Sco 2024 = PNV J17261813-3809354\r\n[[Alert Notice 849 https://www.aavso.org/aavso-alert-notice-849]]', 'priority': True, 'last_data_point': 1738828996, 'observability_times': [['TARGET_RISES', 1745924128]], 'solar_conjunction': False}

    targets = get_targets()
    # convert the dict to a pandas dataframe
    df = pd.DataFrame.from_dict(targets['targets'])   
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
        print(f"Error: {e}")
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

def event_tonight(ephemeris):
    # get the current time
    now = datetime.datetime.now()
    # get the current time in UTC
    now = now.replace(tzinfo=ZoneInfo('UTC'))
    # check there is an element in the ephemeris list that occurs between sunset and sunrise
    # get the sunset and sunrise times
    sunset, sunrise = get_horizon(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # check if the ephemeris is between sunset and sunrise
    #print (ephemeris)
    for ep in ephemeris:
        # check if the ephemeris is between sunset and sunrise
        if ep.timestamp() > sunset and ep.timestamp() < sunrise:
            return ep
    return None