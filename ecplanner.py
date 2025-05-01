from PES_secrets import apikey
import datetime
import requests
import ephem
from corePlanner import check_auth, get_sun_rise_set, get_targets, parse_ephemeris, event_tonight



if __name__ == "__main__":
    # check if we can authenticate with the AAVSO API
    resp = check_auth()
    # check if resp is None
    if resp is None:
        print("Error: Unable to authenticate with the AAVSO API")
        print("Please check your configuration and try again.")
        exit(1)
    # get the current date and time
    datestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    targetdf = get_targets()
    # while testing cut the list down to 10
    targetdf = targetdf.head(10)
    # get the ephemeris for each target in the targetdf
    targetdf['ephemeris'] = None
    for index, row in targetdf.iterrows():
        # check if the ephemeris is empty
        if row['other_info'] is None:
            continue
        # add the ephemeris to the targetdf
        targetdf.at[index, 'ephemeris'] = parse_ephemeris(row['other_info'])
        # update the progress bar by overwriting the line
        print(f"\033[92m\rExamining Ephemeridies: {index+1}/{len(targetdf)}\033[0m", end="")
    print()

    # check if there is an event tonight
    targetdf = event_tonight(targetdf)

    print(targetdf)