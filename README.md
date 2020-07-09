# Imagesim

Imagesim is a classification model leveraging [cloverleaf]() and [pytorch-lightning]() in order to judge the similarity of satellite imagery, inspired by the likes of [terrapattern](http://www.terrapattern.com/) and [GeoVisual Search](https://medium.com/descarteslabs-team/geovisual-search-using-computer-vision-to-explore-the-earth-275d970c60cf).

## Dependencies

* Some standard unix cli tools are required: specifically, `ssh` and `rsync`
* `python` -- version at least 3.7 -- with `pipenv` installed

After those are good to go, install the required python packages with `pipenv install -d --skip-lock`.
Lastly, set up conductor with the command `pipenv run conductor init`.

### Training

Training with the conductor tool is simple: from within a pipenv environment (accessed via `pipenv shell`, run
```bash
# bootstrap an aws instance
$ conductor provision

# train on that instance
$ conductor train

# stop the instance
$ conductor terminate
```
