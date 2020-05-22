import timeit
import numpy as np
import pandas as pd
import json
import orjson
import binascii
import rapidjson #type:ignore
import nujson #type:ignore

loops=1000

rng = np.random.default_rng()
# ndarray int32 because orjson
xnp = rng.integers(1023,size=1000,dtype=np.uint32)
ynp = rng.integers(1023,size=1000,dtype=np.uint32)
x = list([x.item() for x in xnp])
y = list([x.item() for x in ynp])


#######################
# list zip json
# 988us (slow!)
def f2() -> str:
    z =z=[{'x':a,'y':b} for (a,b) in zip(x,y)]
    zz=[{'label':'a', 'data':z}, {'label':'b', 'data':z}]
    return json.dumps(zz)
t = timeit.timeit(f2,number=loops)
print(f'f2 list zip json {1e6*t/loops} us')

#######################
# list zip f
# 724 us even simple operations are slow
def f1() -> str:
    z = [{'x':a,'y':b} for (a,b) in zip(x,y)]
    return f"[{{'label':'a', 'data':{z} }}, {{'label':'b', 'data':{z} }}]"
t = timeit.timeit(f1,number=loops)
print(f'f1 list zip f {1e6*t/loops} us')

#######################
# list zip orjson
# 212 us orjson much faster than f
def f3() -> bytes:
    z = [{'x':a,'y':b} for (a,b) in zip(x,y)]
    zz=[{'label':'a', 'data':z}, {'label':'b', 'data':z}]
    return orjson.dumps(zz)
t = timeit.timeit(f3,number=loops)
print(f'f3 list zip orjson {1e6*t/loops} us')

########################
# list f, parallel lists
# 201 us avoid zip, save 500us
def f0() -> str:
    return f"[{{'label':'a', 'x':{x}, 'y':{y} }}, {{'label':'b', 'x':{x}, 'y':{y} }}]"
t = timeit.timeit(f0,number=loops)
print(f'f0 list f parallel {1e6*t/loops} us')

#######################
# just tobytes, needs encoded
# 116 us, avoid integer serialization
def f4() -> str:
    xnpb = binascii.hexlify(xnp.tobytes())
    ynpb = binascii.hexlify(ynp.tobytes())
    return f"[{{'label':'a', 'x':{xnpb!r}, 'y':{ynpb!r} }}, {{'label':'b', 'x':{xnpb!r}, 'y':{ynpb!r} }}]"
t = timeit.timeit(f4,number=loops)
print(f'f4 ndarray tobytes f {1e6*t/loops} us')

#######################
# zip orjson ndarray -- orjson can't handle numpy types outside numpy arrays
# 60us
#def f5() -> str:
#    z = [{'x':a,'y':b} for (a,b) in zip(xnp,ynp)]
#    zz=[{'label':'a', 'data':z}, {'label':'b', 'data':z}]
#    return orjson.dumps(zz, option=orjson.OPT_SERIALIZE_NUMPY)
#t = timeit.timeit(f5,number=loops)
#print(f'f5 ndarray zip orjson {1e6*t/loops} us')


#####################
# parallel list orjson
# 57 us avoid zip, save 150 us
def f7() -> bytes:
    zz = [{'label':'a', 'x':x, 'y':y }, {'label':'b', 'x':x, 'y':y }]
    return orjson.dumps(zz, option=orjson.OPT_SERIALIZE_NUMPY)
t = timeit.timeit(f7,number=loops)
print(f'f7 parallel list orjson {1e6*t/loops} us')

#######################
# parallel orjson ndarray
# 28 us (fastest), simplest is best
def f6() -> bytes:
    zz = [{'label':'a', 'x':xnp, 'y':ynp }, {'label':'b', 'x':xnp, 'y':ynp }]
    return orjson.dumps(zz, option=orjson.OPT_SERIALIZE_NUMPY)
t = timeit.timeit(f6,number=loops)
print(f'f6 parallel ndarray orjson {1e6*t/loops} us')

#######################
# parallel json ndarray tolist
# 
def f8() -> bytes:
    zz = [{'label':'a', 'x':xnp.tolist(), 'y':ynp.tolist() },
          {'label':'b', 'x':xnp.tolist(), 'y':ynp.tolist() }]
    return json.dumps(zz)
t = timeit.timeit(f8,number=loops)
print(f'f8 parallel ndarray tolist json {1e6*t/loops} us')

#######################
# parallel rapidjson ndarray tolist
# 28 us (fastest), simplest is best
def f9() -> bytes:
    zz = [{'label':'a', 'x':xnp.tolist(), 'y':ynp.tolist() },
          {'label':'b', 'x':xnp.tolist(), 'y':ynp.tolist() }]
    return rapidjson.dumps(zz)
t = timeit.timeit(f9,number=loops)
print(f'f9 parallel ndarray tolist rapidjson {1e6*t/loops} us')

#######################
# parallel nujson ndarray
# 28 us (fastest), simplest is best
def f10() -> bytes:
    zz = [{'label':'a', 'x':xnp, 'y':ynp }, {'label':'b', 'x':xnp, 'y':ynp }]
    return nujson.dumps(zz)
t = timeit.timeit(f10,number=loops)
print(f'f10 parallel ndarray nujson {1e6*t/loops} us')

#####################################################
#                                                   #
# DATAFRAME                                         #
#                                                   #
#####################################################

df = pd.DataFrame()
df['x'] = xnp
df['y'] = ynp
df['load'] = "foo"
df2 = pd.DataFrame()
df2['x'] = xnp
df2['y'] = ynp
df2['load'] = "bar"
df = df.append(df2)
#print(df)

#######################
# dataframe to_records tolist json
# 1700 us
def f11() -> bytes:
    return json.dumps(df.to_records().tolist())
t = timeit.timeit(f11,number=loops)
print(f'f11 dataframe to_records tolist json {1e6*t/loops} us')
#######################
# df column values tolist json
# 699 us
def f12() -> bytes:
    return json.dumps( [
        df.index.values.tolist(),
        df['load'].values.tolist(),
        df['x'].values.tolist(),
        df['y'].values.tolist()
    ])
t = timeit.timeit(f12,number=loops)
print(f'f12 df column values tolist json {1e6*t/loops} us')
#######################
# df slice by load
# 4388 us !! much too slow
def f13() -> bytes:
    loads = df['load'].unique()
    result = {}
    for load in loads:
        loadslice = df[df['load'] == load] 
        result[load + ' x'] = loadslice[['x']].to_records().tolist()
        result[load + ' y'] = loadslice[['y']].to_records().tolist()
    jsonresult = json.dumps(result)
    #print(jsonresult)
    return jsonresult
t = timeit.timeit(f13,number=loops)
print(f'f13 df slice by load {1e6*t/loops} us')
#######################
# df reset_index to_json
# 
def f14() -> bytes:
    return df.reset_index().to_json()
t = timeit.timeit(f14,number=loops)
print(f'f14 df to_json {1e6*t/loops} us')
#######################
# df reset index to json orient records
# 
def f15() -> bytes:
    return df.reset_index().to_json(orient='records')
t = timeit.timeit(f15,number=loops)
print(f'f15 df reset index to json orient records {1e6*t/loops} us')
