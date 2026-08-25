# vendor/

Third-party code copied in so Kaggle needs no extra install.

## termplot

From https://github.com/evanwang810/termplot, MIT, by evanwang810.

Vendored rather than pip-installed for two reasons. Kaggle notebooks would need
another network install that can fail, and an unrelated package called
`termplot` already exists on PyPI, so `pip install termplot` fetches the wrong
thing entirely.

`monitor.py` adds this directory to `sys.path` as a fallback, so a real
`termplot` installed in the environment still wins. To refresh the copy:

```bash
cp -r ../termplot/src/termplot vendor/termplot
```
