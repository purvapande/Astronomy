# astronomy

Sidereal (Lahiri ayanamsa) astronomy on the [Swiss Ephemeris](https://www.astro.com/swisseph/):

- **`astronomy.ephemeris`**: geocentric planet positions and speeds
  (`get_planet_position`), lagna / ascendant (`calculate_lagna`), sunrise,
  sunset, moonrise and moonset (Hindu rising: disc centre, no refraction),
  and eclipse search (`get_eclipses`).
- **`astronomy.eclipse`**: `Eclipse(start, end, place, "solar" | "lunar" | "both")`
  returns each eclipse's contact times, magnitude, local visibility and Sutak
  timings.

## Install

```bash
pip install git+https://github.com/purvapande/astronomy.git
```

Requires Python 3.10+, `pyswisseph` and `pytz`.

## Example

```python
import swisseph as swe
from astronomy.ephemeris import get_planet_position, get_sunrise

jd = swe.julday(2026, 9, 23, 6.5)                            # Julian day, UT
moon_longitude = get_planet_position(jd, 19.076, 72.8777, swe.MOON)
sunrise_jd = get_sunrise(jd, 19.076, 72.8777)
```

## Position cache hook

`get_planet_position` computes every position unless a cache is registered:

```python
from astronomy import ephemeris
ephemeris.set_position_cache(my_cache)   # None removes it
```

`my_cache` is any object with `lookup(planet, jd)`, returning
`(longitude, latitude, speed)` or `None`, and `store(planet, jd, values)`.
Positions are geocentric, so the observer's latitude and longitude are not part
of the key.

## Logging

Both modules log to the standard `logging` logger named `muhurat`. They add no
handlers; configure that logger in your application.

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

If you run a modified version of this code as a network service, the AGPL
requires you to offer its users the corresponding source code.
