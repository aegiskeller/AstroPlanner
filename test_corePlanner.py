from corePlanner import check_auth, get_horizon, get_targets
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