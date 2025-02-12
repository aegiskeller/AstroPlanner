import requests
import ephem
import PES_secrets
from zoneinfo import ZoneInfo

def check_auth():
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/telescope",auth=(PES_secrets.apikey,"api_token"))
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

    print(f"Sunset: {sunset}")
    print(f"Sunrise: {sunrise}")

    zone = ZoneInfo('GMT')
    return(ephem.to_timezone(sunset, zone).timestamp(), 
           ephem.to_timezone(sunrise, zone).timestamp())

def get_targets():
    obs_sections = ['eb'] #ac,ep,cv,eb,spp,lpv,yso,het,misc,all.
    entry = requests.get("https://targettool.aavso.org/TargetTool/api/v1/targets",\
                     auth=(PES_secrets.apikey,"api_token"),params={'observable':['True'],'obs_section':[obs_sections]})
    return(entry.json())