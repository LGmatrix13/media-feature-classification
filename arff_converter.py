from __future__ import annotations

import pandas as pd
from enum import Enum
from typing import NamedTuple, Any
from io import TextIOWrapper

import json
from argparse import Namespace, ArgumentParser

def main(args: Namespace):
    # read the CSV file as a DataFrame
    df = pd.read_parquet(args.src)
    # read the config file as a json file and parse it as an ArffConfig
    with open(args.config, 'rt', encoding='utf-8') as fin:
        config = ArffConfig.from_dict(json.load(fin))
    # use these inputs to produce the requested ARFF file
    dataframe_as_arff(df, config, args.dst)

class ArffConfig(NamedTuple):
    relation: str
    attributes: list[ArffAttributeConfig]
    @staticmethod
    def from_dict(d: dict[str,str|dict[str,str]]) -> ArffConfig:
        # check that required attributes have valid formats
        if not isinstance(d['relation'], str):
            raise ValueError(f"Invalid Relation Name: {d['relation']}")
        if not isinstance(d['attributes'], dict):
            raise ValueError("Invalid Attribute Dictionary")
        return ArffConfig(
            d['relation'],
            [ArffAttributeConfig.from_strings(name, datatype) for name,datatype in d['attributes'].items()]
        )

class ArffAttributeConfig(NamedTuple):
    name: str
    type: ArffDataType
    @staticmethod
    def from_strings(name: str, datatype: str) -> ArffAttributeConfig:
        return ArffAttributeConfig(
            name,
            ArffDataType.from_str(datatype)
        )

class ArffDataType(Enum):
    NUMERIC = 0
    NOMINAL = 1
    STRING  = 2
    @classmethod
    def from_str(cls, datatype: str) -> ArffDataType:
        match datatype.upper():
            case 'NUMERIC': return cls.NUMERIC
            case 'NOMINAL': return cls.NOMINAL
            case 'STRING':  return cls.STRING
            case _: raise ValueError(f"Unknown ARFF Data Type: {datatype}")

def dataframe_as_arff(df: pd.DataFrame, config: ArffConfig, out_path: str):
    # identify the configured names and reduce the dataframe to these attributes
    attribute_names: list[str] = [attr_conf.name for attr_conf in config.attributes]
    df = df[attribute_names]
    # convert columns according to arff config
    for attr_config in config.attributes:
        if attr_config.type == ArffDataType.NUMERIC:
            df[attr_config.name] = [float(x) for x in df[attr_config.name]]
        else:
            df[attr_config.name] = [as_str(x).strip() for x in df[attr_config.name]]
    # clean all column names to be valid for ARFF
    for attr_config in config.attributes:
        df.rename(columns={attr_config.name: clean_col_name(attr_config.name)}, inplace=True)
    
    # open the output file for writing
    with open(out_path, 'wb') as fout:
        # create an alternate writer to use when writing strings directly
        str_fout = TextIOWrapper(fout, encoding='utf-8')
        # write out the relation header line
        str_fout.write(f"@RELATION {config.relation.replace(' ','_')}\n\n")
        # write out one line per attribute in the attribute names list
        for attr_config in config.attributes:
            # write out the prefix with attribute annotation and name
            str_fout.write(f"@ATTRIBUTE {clean_col_name(attr_config.name)} ")
            # write out the attribute datatype according to the specific type of this attribute
            match attr_config.type:
                case ArffDataType.NUMERIC: str_fout.write("NUMERIC\n")
                case ArffDataType.STRING:  str_fout.write("STRING\n")
                case ArffDataType.NOMINAL: str_fout.write('{' + ','.join(map(as_str,set(df[clean_col_name(attr_config.name)]))) + '}\n')
        # write out an extra newline and mark the beginning of data with the @DATA tag
        str_fout.write('\n@DATA\n')
        # flush the output from the buffer before writing the DataFrame data
        str_fout.flush()
        # write out the DataFrame in CSV format
        df.to_csv(fout, index=False, header=False)

def clean_col_name(s: str) -> str:
    return s.replace(' ','_').replace("'",'')

def as_str(x: str|float|int) -> str:
    if is_int(x): return f'_{x}'
    if isinstance(x, str): return x
    else: return str(x)

def is_int(x: Any) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False

if __name__=='__main__':
    parser = ArgumentParser(description=(
        "This command line utility allows for conversion from CSV to ARFF using "
        "a special JSON configuration file to identify the types of each of the "
        "variables as either numeric or nominal."
    ))
    parser.add_argument('src', type=str, help='path to the input CSV file')
    parser.add_argument('config', type=str, help='path to the JSON config file')
    parser.add_argument('dst', type=str, help='path to output arff file')
    args = parser.parse_args()
    main(args)
