from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
from mcstasscript.helper.mcstas_objects import Component as ScriptComponent


def ensure_user_var(instrument: ScriptInstrument, dtype: str, name: str, description: str):
    types_names = [(uv.type, uv.name) for uv in instrument.user_var_list]
    if (dtype, name) in types_names:
        return
    try:
        instrument.user_var(dtype, name=name, comment=description)
    except NameError as err:
        if name in [n for _, n in types_names]:
            print(f"A USERVARS variable with name {name} but type different than {dtype} has already been define.")
        print(f"A variable named {name} exists but is not a USERVARS")


def declare_array(instrument: ScriptInstrument, dtype: str, name: str, description: str, values):
    try:
        instrument.declare(dtype, name=name, comment=description, array=len(values), value=values)
    except NameError:
        print(f"Failed to create a declare variable named {name} -- does it already exist?")