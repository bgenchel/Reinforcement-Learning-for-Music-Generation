import argparse
import xml.etree.ElementTree as ET
import os
import os.path as op
import json
from pathlib import Path


def xml_to_dict(fpath):
    def recurse(root, root_dict):
        if root.tag == "part":
            root_dict["measures"] = []
        if root.tag == "measure":
            root_dict["groups"] = []
        if root.tag == "harmony":
            root_dict["degrees"] = []

        for i, child in enumerate(root):
            child_dict = {'attributes': child.attrib}
            if len(list(child)) == 0:
                child_dict['text'] = child.text
            else:
                child_dict = recurse(child, child_dict)

            if child.tag == "measure":
                child_dict["harmonies_start"] = harmony_timing(child)
                root_dict["measures"].append(child_dict)
            elif child.tag == "harmony":
                root_dict["groups"].append({"harmony": child_dict, "notes": []})
            elif child.tag == "degree":
                root_dict["degrees"].append(child_dict)
            elif child.tag == "note":
                if not len(root_dict["groups"]):
                    root_dict["groups"].append({"harmony": {}, "notes": []})
                root_dict["groups"][-1]["notes"].append(child_dict)
            elif child.tag == "direction":
                if "words" in child_dict["direction-type"].keys():
                    root_dict["words"] = child_dict["direction-type"]["words"] 
                elif "rehearsal" in child_dict["direction-type"].keys():
                    root_dict["rehearsal"] = child_dict["direction-type"]["rehearsal"]
            else:
                root_dict[child.tag] = child_dict
        return root_dict

    xml_dict = {}
    tree = ET.parse(fpath)
    root = tree.getroot()
    xml_dict[root.tag] = {"attributes": root.attrib}
    return recurse(root, xml_dict)

# Used to get the start position of a chord for use in MIDI tick representation
def harmony_timing(measure):
    harmonies_start = []

    position = 0

    for el in measure:
        # Advance position for note
        if el.tag == "note":
            for child in el:
                if child.tag == "duration":
                    position += int(child.text)
        elif el.tag == "forward":
            # Forward has one child, duration
            position += int(el[0].text)
        elif el.tag == "backup":
            # Backward has one child, duration
            position -= int(el[0].text)
        elif el.tag == "harmony":
            # Capture the start time of this harmony
            harmonies_start.append(position)

    return harmonies_start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=("charlie_parker", "bebop", "nottingham"), type=str,
                        required=True, help="the dataset to convert")
    args = parser.parse_args()

    root_dir = str(Path(op.abspath(__file__)).parents[3])
    xml_path = op.join(root_dir, 'data', 'raw', args.dataset + '-xml')
    if not op.exists(xml_path):
        raise Exception("no xml directory exists.")

    json_path = op.join(root_dir, 'data', 'interim', args.dataset + '-json')
    if not op.exists(json_path):
        os.makedirs(json_path)

    fpaths = [op.join(xml_path, fname) for fname in os.listdir(xml_path)]
    for fpath in fpaths:
        print('parsing %s...' % op.basename(fpath))
        try:
            if op.basename(fpath)[-4:] == ".xml":
                with open(op.join(json_path, op.basename(fpath).split('.')[0] + '.json'), 'w') as fp:
                    json.dump(xml_to_dict(fpath), fp, indent=4)
            else:
                raise Exception()
        except:
            print('Error!')
