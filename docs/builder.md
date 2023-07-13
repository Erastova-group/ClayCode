# ClayCode.builder 

Builds clay models based on average target occupancies for a specified unit cell type.

`ClayCode.builder` will match a combination of differently substituted unit cells to fit the average target occupancies specified by the user.


## Usage

Arguments for `builder` are:

* `-f`: [System specifications YAML file](YAML.md) (include .csv file path)
* `-comp`: [Clay composition in CSV format](CSV.md) (can also be given in system specifications YAML, see [Pyrophyllite Tutorial](pyr.md))
* `--manual_setup`: (run builder in interactive mode)


### Example:

```shell
ClayCode builder -f path/to/input_Clay.yaml
```



