Builder 

Builds clay models based on average target occupancies for a specified unit cell type.

`ClayCode.builder` will match a combination of differently substituted unit cells to fit the average target occupancies specified by the user.

### Usage

Arguments for `builder` are:
  - `-f`: [System specifications YAML file] (docs/)
  - `-comp`: [Clay composition in CSV format](#2-clay-composition-in-csv-format) (can also be given in system specifications YAML)
  - `--manual_setup` (run builder in interactive mode)
#### Example:

```shell
ClayCode builder -f path/to/input_NAu-1-fe.yaml
```

### Required input files:

* YAML
* CSV