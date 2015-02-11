Computational Geometry
======================

The code runs from the unix shell.  To run the code from the unix terminal,
install the items from requirements.txt, then use the following command 

```
./shapes_1.2.py [file name]
```

make sure the file you want to use is in the same directory as the code!

For example if you run the following command from inside the attached zip: 

```
./shapes_1.2.py shapes.json
```

you should see:

'''
'odd' is not a polygon
'square' surrounds 'triangle'
'square' intersects 'rectangle'
'square' is separate from 'pentagon'
'triangle' is inside 'square'
'triangle' intersects 'rectangle'
'triangle' is separate from 'pentagon'
'rectangle' intersects 'square'
'rectangle' intersects 'triangle'
'rectangle' is separate from 'pentagon'
'pentagon' is separate from 'square'
'pentagon' is separate from 'triangle'
'pentagon' is separate from 'rectangle'
'''


You may have to change permissions on the code file, if so run the following command:

```
chmod 755 shapes_1.2.py
```

then run:

```
./shapes_1.2.py [file name]
```