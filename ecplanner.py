from PES_secrets import apikey
import datetime
import requests
import ephem
from corePlanner import get_horizon, parse_targets, parse_ephemeris, event_tonight



if __name__ == "__main__":
    # generate a datstr for the current date
    # or use a specific date string
    # Example: "2023-10-01 00:00:00"
    # Note: The date string should be in the format "YYYY-MM-DD HH:MM:SS"
    # Example usage
    datestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#    print(get_horizon(datestr))
    # print(get_targets())
    targetdf = parse_targets()  
#    print(targetdf)
#    print(targetdf.columns)
#    print(targetdf['other_info'].iloc[95])
    # get the ephemeris for each target in the targetdf
    # generate a progress bar
    # for each target in the targetdf
    targetdf['ephemeris'] = None
    for index, row in targetdf.iterrows():
        # check if the ephemeris is empty
        if row['other_info'] is None:
            continue
        # parse the ephemeris from the other_info column
        #print(row['other_info'])
        #print(parse_ephemeris(row['other_info']))
        # add the ephemeris to the targetdf
        targetdf.at[index, 'ephemeris'] = parse_ephemeris(row['other_info'])
        # update the progress bar by overwriting the line
        print(f"\033[92m\rExamining Ephemeridies: {index+1}/{len(targetdf)}\033[0m", end="")
    print()

    # for each target check if there is an event tonight
    for index, row in targetdf.iterrows():
        # check if the ephemeris is empty
        if row['ephemeris'] is None:
            continue
        # check if the ephemeris is between sunset and sunrise
        if event_tonight(row['ephemeris']):
            # print the line in green
            print(f"\033[92mEvent tonight for {row['star_name']}\033[0m")
        else:
            print(f"No event tonight for {row['star_name']}")