
Pyrophyllite is a phyllosilicate mineral composed of aluminium silicate hydroxide: Al2Si4O10(OH)2.


non-charged

can be with Fe-incorporated, example:
https://hal.science/insu-00160523/

or

Pütürge pyrophyllite with small iron and chromium contents has a bright white colour after firing, and it is has been used in white cement production, where it is known to produce the whitest cement in Europe (Uygun & Solakoğlu, Reference Uygun and Solakoğlu2002). 

https://www.cambridge.org/core/journals/clay-minerals/article/beneficiation-of-the-puturge-pyrophyllite-ore-by-flotation-mineralogical-and-chemical-evaluation/2E4E88D5CF9A97EAFCE9E6997E2AB1DA

There is no .CSV file, as quite simple set up, using 1 (or 2 if incl. some Fe) UCs:


from the D21_UCs_infor.csv

```

UC,st,at,fet,ao,feo,fe2,mgo,cao,lio,mgh,cah,C_Td,C_Oct,Comments
D221,8,0,0,4,0,0,0,0,0,0,0,0,0,Smectite UC neutral
D228,8,0,0,3,1,0,0,0,0,0,0,0,0,Smectite UC neutral Oct Fe3

```


The .yaml file can set up with something like this: 

```yaml
# required UCs to builder system as list
UC_INDEX_LIST: [D221 D228]

# probability list of unit cells in system
UC_RATIOS_LIST: [0.9 0.1]
```


```shell
ClayCode builder -f Tutorial/Pyr.yaml
```


