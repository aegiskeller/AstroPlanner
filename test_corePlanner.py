from corePlanner import check_auth, get_horizon, get_targets, parse_targets, parse_ephemeris
import datetime


def test_auth_toaavso():
    assert isinstance(check_auth(), dict), "Result is not a dictionary"
    assert -35.351835 in check_auth().values(), "-35.351835 not found in the result values"

def test_get_horizon():
    ss, sr  = get_horizon(datetime.datetime.today().strftime('%Y-%m-%d'))
    assert isinstance(ss, float), "Sunset is not a float"
    assert isinstance(sr, float), "Sunrise is not a float"

def test_get_targets():
    assert isinstance(get_targets(), dict), "Result is not a dictionary"

def test_parse_targets():
    # Assuming parse_targets() returns a DataFrame
    targetdf = parse_targets()
    assert isinstance(targetdf, pd.DataFrame), "Result is not a DataFrame"
    assert not targetdf.empty, "DataFrame is empty"
    assert 'other_info' in targetdf.columns, "'other_info' column not found in DataFrame"
    assert len(targetdf['other_info']) > 0, "'other_info' column is empty"

def test_parse_ephemeris():
    ref = [['Ephemeris info https://www.aavso.org/vsx/index.php?view=detail.ephemeris&nolayout=1&oid=469680']]
    ephemeris = parse_ephemeris(ref)
    assert isinstance(ephemeris, str), "Result is not a string"