import msgspec
import gzip
import json

import ipdb
st=ipdb.set_trace

VISER_PATH = "/data0/jhshao/data/bear_animation.viser"
OUTPUT_DIR = "data/recording_bear.json"

if __name__ == "__main__":
    with gzip.open(VISER_PATH, "rb") as f:
        viser_record = f.read()
    viser = msgspec.msgpack.decode(viser_record)
    st()
    viser['messages'] = viser['messages'][1318:]
    with open(OUTPUT_DIR, "w") as f:
        json.dump(str(viser), f, indent=4)
    
    
