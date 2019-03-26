import json
import math
import numpy as np
import os.path as op
import os
import sys
from pathlib import Path
from harmony import Harmony, ROOT_DICT
from copy import deepcopy
import pickle
import argparse

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const

TICKS_PER_BEAT = 24 # 96 ticks per bar in 4/4
MIDI_RANGE = 128

MAX_ROOT_TOKEN = max(ROOT_DICT.values())


def rotate(l, x):
    """
    Rotates a list a given number of steps
    :param l: the list to rotate
    :param x: a positive or negative integer steps to rotate
    :return: the rotated list
    """
    # print(type(x))
    return l[-x:] + l[:-x]


def shift_root_token(rt, steps):
    """
    the rules are as follows:
        - if the root_token is 0, meaning no chord, don't do anything.
        - if the root_token is 12, shift forward goes to 1
        - if the root_token is 1, shift backward goes to 12
    """
    if rt == 0:
        return rt

    if steps > 0:
        new_rt = rt + steps
        if new_rt > MAX_ROOT_TOKEN:
            new_rt = new_rt % MAX_ROOT_TOKEN
    elif steps < 0:
        new_rt = rt - steps
        if new_rt == 0:
            new_rt = MAX_ROOT_TOKEN
        elif new_rt < 0:
            new_rt = MAX_ROOT_TOKEN + new_rt

    return rt


class Parser:
    def __init__(self, root_dir=None, json_dir=None, out_dir=None, dataset=None):
        """
        Loads, parses, and formats JSON data into model-ready inputs.
        :param output: the desired parsing format, either "pitch_duration_tokens" or "midi_ticks"
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param out_dir: the directory to save parsed individual songs
        """
        self.root_dir, \
        self.json_dir, \
        self.out_dir = self.verify_directories(root_dir, json_dir, out_dir, dataset)

        # Individual file names
        self.json_paths = [op.join(self.json_dir, filename) for filename in os.listdir(self.json_dir)]

        # Storage for the parsed output
        self.parsed = None

    def parse(self):
        pass

    @staticmethod
    def verify_directories(root_dir, json_dir, out_dir, dataset):
        """
        Ensures that all input/output directories exist.
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param out_dir: the directory to save parsed individual songs
        :return:
        """
        # Directory of the project, from which to base other dirs
        if not root_dir:
            # Looks up 3 directories to get project dir
            root_dir = str(Path(op.abspath(__file__)).parents[3])

        # Directory where JSON files to be parsed live
        if not json_dir:
            json_dir = op.join(root_dir, 'data', 'interim', dataset + '-json')

        if not op.exists(json_dir):
            raise Exception("JSON directory {} not found.".format(json_dir))

        # Directory where individual parsed songs get saved to
        if not out_dir:
            out_dir = op.join(root_dir, 'data', 'interim', dataset + '-parsed')

        if not op.exists(out_dir):
            os.makedirs(out_dir)

        return root_dir, json_dir, out_dir

    @classmethod
    def parse_metadata(cls, filename, song_dict):
        """
        Given the JSON input as a dict, returns.
        :param filename: name of the file from which song_dict is loaded, used as a backup for title/artist
        :param song_dict:  a song dict in the format created by src/processing/conversion/xml_to_json.py
        :return: an object containing metadata for the song in the following format:
        {
          title,
          artist,
          key,
          time_signature
        }
        """
        # Strip filename of path and file extension (.json)
        filename = filename.split('/')[-1][:-5]

        # Key
        key, multiple = cls.get_key(song_dict)
        if multiple:
            # Disregard songs with multiple keys
            return None

        # Title
        if "movement-title" in song_dict:
            title = song_dict["movement-title"]["text"]
        elif "work" in song_dict:
            if "work-title" in song_dict["work"]:
                title = song_dict["work"]["work-title"]["text"]
        else:
            title = filename.split('-')[1]

        # Artist
        if ("identification" in song_dict) and ("creator" in song_dict["identification"]):
            artist = song_dict["identification"]["creator"]["text"]
        else:
            artist = filename.split('-')[0]

        # Time signature
        time_dict = song_dict["part"]["measures"][0]["attributes"]["time"]
        time_signature = "%s/%s" % (time_dict["beats"]["text"], time_dict["beat-type"]["text"])

        return {
            "title": title,
            "artist": artist,
            "key": key,
            "time_signature": time_signature
        }

    @staticmethod
    def get_key(song_dict):
        """
        Fetches a key from the JSON representation of MusicXML for a song.

        I know from analysis that the only keys in my particular dataset are major
        and minor.
        :param song_dict: a song dict in the format created by src/processing/conversion/xml_to_json.py
        :return: ff there's only one key: (key, False), if there's more than one key: (None, True)
        """
        key = None
        multiple = False

        # Check each measure to ensure there isn't a key change, if so, return
        for measure in song_dict["part"]["measures"]:
            if "key" in measure["attributes"].keys():
                if key is not None:
                    multiple = True
                    return key, multiple

        # Get key from first measure
        key_dict = song_dict["part"]["measures"][0]["attributes"]["key"]
        position = key_dict["fifths"]["text"]
        if "mode" in key_dict.keys():
            mode = key_dict["mode"]["text"]
        else:
            mode = "major"  # just assume, it doesn't really matter anyways
        try:
            key = "%s%s" % (const.KEYS_DICT[mode][position], mode)
        except KeyError:
            print("Error!! mode: {}, position: {}".format(mode, position))
            key = None

        return key, multiple

    @staticmethod
    def get_divisions(song_dict):
        """
        Fetch the divisions per quarter note in the JSON represented MusicXML
        :param song_dict: a song dict in the format created by src/processing/conversion/xml_to_json.py
        :return: the number of divisions a quarter note is split into in this song's MusicXML representation
        """
        return int(song_dict["part"]["measures"][0]["attributes"]["divisions"]["text"])


class TickParser(Parser):

    def __init__(self, root_dir=None, json_dir=None, out_dir=None, dataset=None):
        """
        Loads, parses, and formats JSON data into model-ready inputs.
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param out_dir: the directory to save parsed individual songs
        """
        super().__init__(root_dir, json_dir, out_dir, dataset)
        self.ticks = TICKS_PER_BEAT
        self.parse()

    def parse(self):
        """
        Parses the JSON formatted MusicXML into MIDI ticks format.
        :return: a list of parsed songs in the following format:
        {
          metadata: {
            title,
            artist,
            key,
            time_signature,
            ticks_per_measure
          },
          measures: [[{
            harmony: {
              root,
              pitch_classes
            },
            ticks (num_ticks x 89) (Ab0 - C8): {}
          }], ...]
        }
        or, None, if a song has more than one key
        """
        songs = []
        skipped = 0
        for filename in self.json_paths:
            # Load song dict
            try:
                song_dict = json.load(open(filename))
            except:
                print("Unable to load JSON for %s" % filename)
                continue

            print("Parsing %s" % op.basename(filename))

            # Get song metadata
            metadata = super().parse_metadata(filename, song_dict)

            # Add ticks per measure to metadata
            time_signature = [int(n) for n in metadata["time_signature"].split("/")]
            metadata["ticks_per_measure"] = int(time_signature[0] * (4 / time_signature[1]) * self.ticks)

            # Get song divisions
            divisions = super().get_divisions(song_dict)

            # Calculate scale factor from divisions to ticks
            #  i.e. scale_factor * divisions = num_ticks
            scale_factor = self.ticks / divisions

            # if scale_factor > 1
            #     skipped += 1
            #     print(skipped)
            #     continue
            # raise Exception("Error: MusicXML has lower resolution than desired MIDI ticks.")

            # Parse each measure
            measures = []

            # Empty harmony for the beginning of the piece
            # last_harmony = {"root": [0 for _ in range(12)],
            #                 "pitch_classes": [0 for _ in range(12)]}
            # Empty harmony for the beginning of the piece
            last_harmony = {"root": 0, "type": 0, "pitch_classes": [0 for _ in range(12)]}
            for measure in song_dict["part"]["measures"]:
                parsed_measure, last_harmony = self.parse_measure(measure, scale_factor, last_harmony)
                measures.append(parsed_measure)

            songs.append({
                "metadata": metadata,
                "measures": measures
            })

        self.parsed = songs

    def parse_measure(self, measure, scale_factor, prev_harmony):
        """
        For a measure, returns a set of ticks grouped by associated harmony in.
        :param measure: a measure dict in the format created by src/processing/conversion/xml_to_json.py
        :param scale_factor: the scale factor between XML divisions and midi ticks
        :param prev_harmony: a reference to the last harmony used in case a measure has none
        :return: a dict containing a list of groups that contains a harmony and the midi ticks associated with that harmony
        """
        parsed_measure = {"groups": []}
        if "rehearsal" in measure.keys():
            parsed_measure["rehearsal"] = ["text"]
        if "words" in measure.keys():
            parsed_measure["words"] = ["text"]

        total_ticks = 0
        for group in measure["groups"]:
            # Set note value for each tick in the measure
            group_ticks = []
            for note in group["notes"]:
                if not "duration" in note:
                    print("Skipping grace note...")
                    continue
                divisions = int(note["duration"]["text"])
                num_ticks = int(scale_factor * divisions)
                index = self.get_note_index(note)

                for i in range(num_ticks):
                    group_ticks.append(index)
                    # tick = [0 for _ in range(MIDI_RANGE)]
                    # tick[index] = 1
                    # group_ticks.append(tick)

            total_ticks += len(group_ticks)

            if not group["harmony"]:
                parsed_measure["groups"].append({"harmony": prev_harmony, "ticks": group_ticks})
            else:
                harmony = Harmony(group["harmony"])
                harmony_dict = {"root": harmony.get_root_token(),
                                "type": harmony.get_type_token(),
                                "pitch_classes": harmony.get_seventh_pitch_classes_binary()}
                prev_harmony = harmony_dict
                parsed_measure["groups"].append({"harmony": harmony_dict, "ticks": group_ticks})

            # Mitigate ticks for chords that occur mid-note
            for i, group in enumerate(parsed_measure["groups"]):
                try:
                    if not group["ticks"]:
                        # Handle the case of no harmony at the start of the bar
                        if not 0 in measure["harmonies_start"]:
                            measure["harmonies_start"].insert(0, 0)

                        correct_len_of_prev_harmony = int(
                            scale_factor * (measure["harmonies_start"][i] - measure["harmonies_start"][i - 1])
                        )

                        group["ticks"].extend(parsed_measure["groups"][i - 1]["ticks"][correct_len_of_prev_harmony:])
                        parsed_measure["groups"][i - 1]["ticks"] = parsed_measure["groups"][i - 1]["ticks"][:correct_len_of_prev_harmony]
                except:
                    import pdb
                    pdb.set_trace()
                    raise ("No ticks in the first group of a measure! (in fix for chords mid-note)")

        if total_ticks > TICKS_PER_BEAT * 4:
            raise Exception("OH NO BRO. YOUR TICKS ARE TOO MUCH YO")

        i = 0
        while total_ticks < TICKS_PER_BEAT * 4:
            group = parsed_measure["groups"][i]
            # spacer_tick = [0 for _ in range(MIDI_RANGE)]
            # spacer_tick[0] = 1  # fill with rests
            spacer_tick = 0
            group["ticks"].append(spacer_tick)
            i = (i + 1) % len(parsed_measure["groups"])
            total_ticks += 1

        parsed_measure["num_ticks"] = total_ticks
        return parsed_measure, prev_harmony

    def save_parsed(self, transpose=False):
        """
        Saves the parsed songs as .pkl to song_dir.
        :param transpose: if True, transposes and saves each song in all 12 keys.
        :return: None
        """
        # Ensure parsing has happened
        if not self.parsed:
            print("Nothing has been parsed.")
            return

        for song in self.parsed:
            if transpose:
                for steps in range(-6, 6):
                    transposed = self.transpose_song(song, steps)
                    filename = "-".join([
                        "_".join(transposed["metadata"]["title"].split(" ")),
                        "_".join(transposed["metadata"]["artist"].split(" "))]) + "_%d" % steps + ".pkl"
                    filename = filename.replace("/", ",")
                    outpath = op.join(self.out_dir, filename)
                    pickle.dump(transposed, open(outpath, 'wb'))
            else:
                filename = "-".join([
                    "_".join(song["metadata"]["title"].split(" ")),
                    "_".join(song["metadata"]["artist"].split(" "))]) + ".pkl"
                filename = filename.replace("/", ",")
                outpath = op.join(self.out_dir, filename)
                pickle.dump(song, open(outpath, 'wb'))

    @staticmethod
    def get_note_index(note):
        """
        Fetches an index value for encoding a note's pitch for MIDI tick pitch formatting.
        :param note: a note dict in the format created by src/processing/conversion/xml_to_json.py
        :return: an index representing a note value, where 0 is F3, 35 is E6, and 36 is 'rest'
        """
        if "rest" in note.keys():
            return 0
        else:
            note_string = note["pitch"]["step"]["text"]
            if "alter" in note["pitch"].keys():
                note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                    note["pitch"]["alter"]["text"])
            octave = int(note["pitch"]["octave"]["text"])
            note_index = (octave + 1) * 12 + const.NOTES_MAP[note_string]
            return note_index

    @classmethod
    def transpose_song(cls, song, steps):
        """
        Transposes a song that has been parsed into MIDI ticks form.
        :param song: a song parsed into MIDI ticks form
        :param steps: a positive or negative number representing how many steps to transpose
        :return: a transposed song in MIDI ticks form
        """
        sign = lambda x: (1, -1)[x < 0]
        transposed = deepcopy(song)
        print("transposing by %i" % steps)
        transposed["transposition"] = steps

        for measure in transposed["measures"]:
            for group in measure["groups"]:
                if group["harmony"]:
                    # Transpose harmony, no need to transpose type
                    group["harmony"]["root"] = shift_root_token(group["harmony"]["root"], steps)
                    group["harmony"]["pitch_classes"] = rotate(group["harmony"]["pitch_classes"], steps)

                # Transpose ticks
                direction = sign(steps)
                new_ticks = group["ticks"]
                for _ in range(abs(steps)):
                    transposed_ticks = []
                    for tick in new_ticks:
                        transposed_ticks.append(cls.transpose_tick(tick, direction))
                    new_ticks = transposed_ticks
                group["ticks"] = new_ticks

        return transposed

    @staticmethod
    def transpose_tick(tick, direction):
        """
        Transposes a MIDI tick array one step up or down.
        :param ticks: a one-hot array representing a pitch F3-E6 or rest
        :param direction: either -1 or 1, representing the direction to transpose
        :return: a transposed MIDI tick array
        """
        # note = tick.index(1)
        note = tick
        # print(tick)

        # Don't have to transpose a rest
        if note == 0:
            return tick

        # Transpose up a step
        if direction > 0:
            if note == MIDI_RANGE - 2:
                # Make E6 transpose "up" to F5
                # transposed = np.zeros((len(tick)))
                # transposed[note - 11] = 1
                # return transposed
                return note - 11
            else:
                # transposed = tick
                # transposed.insert(0, transposed.pop(len(transposed) - 2))
                # return transposed
                return note + 1
        else:
            # Transpose down a step
            if note == 1:
                # Make F3 transpose "down" to E4
                # transposed = [0 for i in range(len(tick))]
                # transposed[note + 11] = 1
                # return transposed
                return note + 11
            else:
                # transposed = tick
                # transposed.insert(len(transposed) - 2, transposed.pop(0))
                # return transposed
                return note - 1


class PitchDurParser(Parser):

    def __init__(self, root_dir=None, json_dir=None, song_dir=None, dataset_dir=None):
        """
        Loads, parses, and formats JSON data into model-ready inputs.
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param song_dir: the directory to save parsed individual songs
        :param dataset_dir: the directory to save full datasets
        """
        super().__init__(root_dir, json_dir, song_dir, dataset_dir)
        self.parse()

    def parse(self):
        """
        Parses the JSON formatted MusicXML into pitch duration tokens format.
        :return: a list of parsed songs in the following format:
        {
          metadata: {
            title,
            artist,
            key,
            time_signature
          },
          measures: [{
            groups: {
              harmony: {
                root,
                pitch_classes
              },
              pitch_numbers,
              duration_tags,
              bar_position
            }, ...
          }, ...]
        }
        or, None, if a song has more than one key
        """
        songs = []

        for filename in self.json_paths:
            # Load song dict
            try:
                song_dict = json.load(open(filename))
            except:
                print("Unable to load JSON for %s" % filename)
                continue

            print("Parsing %s" % op.basename(filename))

            # Get song metadata
            metadata = super().parse_metadata(filename, song_dict)

            # Get song divisions
            divisions = super().get_divisions(song_dict)

            # Parse each measure
            measures = []
            for measure in song_dict["part"]["measures"]:
                parsed_measure = self.parse_measure(measure, divisions)
                measures.append(parsed_measure)

            # try to fill in harmonies somewhat naively
            max_harmonies_per_measure = 0
            for i, measure in enumerate(measures):
                for j, group in enumerate(measure["groups"]):
                    if not group["harmony"]:
                        if i == 0 and j == 0:
                            for after_group in measures[i]["groups"][j + 1:]:
                                if after_group["harmony"]:
                                    measure["groups"][j]["harmony"] = after_group["harmony"]
                                    break
                            for after_measure in measures[i + 1:]:
                                for after_measure_group in after_measure["groups"]:
                                    if after_measure_group["harmony"]:
                                        measure["groups"][j]["harmony"] = after_measure_group["harmony"]
                                        break
                        elif i == 0:
                            for before_group in measure["groups"][j - 1::-1]:
                                if before_group["harmony"]:
                                    measure["groups"][j]["harmony"] = before_group["harmony"]
                                    break
                        else:
                            for before_group in measure["groups"][j - 1::-1]:
                                if before_group["harmony"]:
                                    measure["groups"][j]["harmony"] = before_group["harmony"]
                                    break
                            for before_measure in measures[i - 1::-1]:
                                for before_measure_group in before_measure["groups"]:
                                    if before_measure_group["harmony"]:
                                        measure["groups"][j]["harmony"] = before_measure_group["harmony"]
                                        break
                max_harmonies_per_measure = max(len(measure["groups"]), max_harmonies_per_measure)

            if max_harmonies_per_measure == 0:
                continue
            
            # add next harmony 
            for i, measure in enumerate(measures):
                for j, group in enumerate(measure["groups"]):
                    group["next_harmony"] = {}
                    if j < len(measure["groups"]) - 1:
                        group["next_harmony"]["root"] = measure["groups"][j + 1]["harmony"]["root"]
                        group["next_harmony"]["pitch_classes"] = measure["groups"][j + 1]["harmony"]["pitch_classes"]
                    elif j == len(measure["groups"]) - 1 and i < len(measures) - 1:
                        group["next_harmony"]["root"] = measures[i + 1]["groups"][0]["harmony"]["root"]
                        group["next_harmony"]["pitch_classes"] = measures[i + 1]["groups"][0]["harmony"]["pitch_classes"]
                    elif j == len(measure["groups"]) - 1 and i == len(measures) - 1:
                        group["next_harmony"]["root"] = measures[0]["groups"][0]["harmony"]["root"]
                        group["next_harmony"]["pitch_classes"] = measures[0]["groups"][0]["harmony"]["pitch_classes"]

            songs.append({
                "metadata": metadata,
                "measures": measures
            })

        self.parsed = songs

    def parse_measure(self, measure, divisions):
        """
        Parses a measure according to the pitch duration token parsing process.
        :param measure: a measure dict in the format created by src/processing/conversion/xml_to_json.py
        :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
        :return: a dict containing a list of groups containing the pitch numbers, durations, and bar positions for each
                harmony in a measure
        """
        parsed_measure = {
            "groups": []
        }

        tick_idx = 0
        for group in measure["groups"]:
            parsed_group = {
                "harmony": {},
                "pitch_numbers": [],
                "duration_tags": [],
                "bar_positions": []
            }
            harmony = Harmony(group["harmony"])
            parsed_group["harmony"]["root"] = harmony.get_one_hot_root()
            parsed_group["harmony"]["pitch_classes"] = harmony.get_type_token()
            # parsed_group["harmony"]["pitch_classes"] = harmony.get_seventh_pitch_classes_binary()
            dur_ticks_list = []
            for note_dict in group["notes"]:
                # want monophonic, so we'll just take the top note
                if "chord" in note_dict.keys() or "grace" in note_dict.keys():
                    continue
                else:
                    pitch_num, dur_tag, dur_ticks = self.parse_note(note_dict, divisions)
                    parsed_group["pitch_numbers"].append(pitch_num)
                    parsed_group["duration_tags"].append(dur_tag)
                    dur_ticks_list.append(dur_ticks)
            unnorm_barpos = [tick_idx + sum(dur_ticks_list[:i]) for i in range(len(dur_ticks_list))]
            bar_positions = [int((dur_ticks / (4 * divisions)) * 96)  for dur_ticks in unnorm_barpos]
            parsed_group["bar_positions"] = bar_positions
            parsed_measure["groups"].append(parsed_group)
            tick_idx += sum(dur_ticks_list)
        return parsed_measure

    def parse_note(self, note_dict, divisions):
        """
        Parses a note according to the pitch duration token parsing process.
        :param note_dict: a note dict in the format created by src/processing/conversion/xml_to_json.py
        :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
        :return: a number reflecting the pitch, a string representing the duration, and the duration in divisions
        """
        if "rest" in note_dict.keys():
            pitch_num = const.NOTES_MAP["rest"]
        elif "pitch" in note_dict.keys():
            note_string = note_dict["pitch"]["step"]["text"]
            if "alter" in note_dict["pitch"].keys():
                note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                    note_dict["pitch"]["alter"]["text"])
            octave = int(note_dict["pitch"]["octave"]["text"])
            pitch_num = (octave - 1) * 12 + 3 + const.NOTES_MAP[note_string]

        dur_tag, dur_ticks = self.get_note_duration(note_dict, divisions)
        return pitch_num, dur_tag, dur_ticks

    def save_parsed(self, transpose=False):
        """
        Saves the parsed songs as .pkl to song_dir.
        :param transpose: if True, transposes and saves each song in all 12 keys.
        :return: None
        """
        # Ensure parsing has happened
        if not self.parsed:
            print("Nothing has been parsed.")
            return

        for song in self.parsed:
            if transpose:
                for steps in range(-6, 6):
                    transposed = self.transpose_song(song, steps)

                    filename = "-".join([
                        "_".join(transposed["metadata"]["title"].split(" ")),
                        "_".join(transposed["metadata"]["artist"].split(" "))]) + "_%d" % steps + ".pkl"
                    filename = filename.replace("/", ",")
                    outpath = op.join(self.song_dir, filename)
                    pickle.dump(transposed, open(outpath, 'wb'))
            else:
                filename = "-".join([
                    "_".join(song["metadata"]["title"].split(" ")),
                    "_".join(song["metadata"]["artist"].split(" "))]) + ".pkl"
                filename = filename.replace("/", ",")
                outpath = op.join(self.out_dir, filename)
                print("dumping to: %s" % outpath)
                pickle.dump(song, open(outpath, 'wb'))

    @staticmethod
    def get_note_duration(note_dict, divisions=24):
        """
        Fetches the duration of a note in JSON formatted MusicXML.
        :param note_dict: a note dict in the format created by src/processing/conversion/xml_to_json.py
        :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
        :return: a string representing the duration, and the duration in divisions
        """
        init_dur_dict = {'double': divisions * 8, 'whole': divisions * 4, 'half': divisions * 2,
                         'quarter': divisions, '8th': divisions / 2, '16th': divisions / 4, '32nd': divisions / 8}

        dur_dict = {}
        for k, v in init_dur_dict.items():
            dur_dict[k] = v
            dur_dict[k + '-triplet'] = (2 * v / 3, math.floor(2 * v / 3), math.ceil(2 * v / 3))
            dur_dict[k + '-dot'] = (3 * v / 2, math.floor(3 * v / 2), math.ceil(3 * v / 2))

        # print('divisions: %d' % divisions)
        # print('dur_dict: {}'.format(dur_dict))
        if "duration" not in note_dict.keys():
            note_dur = -1
        else:
            note_dur = float(note_dict["duration"]["text"])

        label = None
        if "type" in note_dict.keys():
            note_type = note_dict["type"]["text"]

            if note_type == "eighth":
                note_type = "8th"

            label = note_type
            dot_dur = 3 * dur_dict[note_type] / 2
            triplet_dur = 2 * dur_dict[note_type] / 3
            if note_dur == dur_dict[note_type]:
                pass
            elif note_dur in dur_dict[note_type + '-dot']:
                label = '-'.join([label, 'dot'])
            elif note_dur in dur_dict[note_type + '-triplet']:
                label = '-'.join([label, 'triplet'])
            else:
                print("Undefined %s duration. Entering as regular %s." % (note_type, note_type))
        else:
            for k, v in dur_dict.items():
                if type(v) == tuple:
                    if note_dur in v:
                        label = k
                        break
                elif note_dur == v:
                    label = k
                    break

        if label is None:
            print("Undefined duration %.2f. Labeling 'other'." % (note_dur / divisions))
            label = "none"

        return const.DURATIONS_MAP[label], note_dur

    @classmethod
    def transpose_song(cls, song, steps):
        """
        Transposes a song in the pitch duration token format.
        :param song: a song in pitch duration token format
        :param steps: positive or negative integer number of steps to transpose
        :return: a transposed song in pitch duration token format
        """
        transposed = deepcopy(song)
        transposed["transposition"] = steps
        print("transposing by %i" % steps)
        for i, measure in enumerate(transposed["measures"]):
            for j, group in enumerate(measure["groups"]):
                group["pitch_numbers"] = [
                    (lambda n: n + steps if n != const.NOTES_MAP["rest"] else n)(pn)
                    for pn in group["pitch_numbers"]]
                group["harmony"]["root"] = rotate(group["harmony"]["root"], steps)
                group["harmony"]["type"] = rotate(group["harmony"]["type"], steps)
                group["harmony"]["pitch_classes"] = rotate(group["harmony"]["pitch_classes"], steps)
                group["next_harmony"]["root"] = rotate(group["next_harmony"]["root"], steps)
                group["next_harmony"]["type"] = rotate(group["next_harmony"]["type"], steps)
                group["next_harmony"]["pitch_classes"] = rotate(group["next_harmony"]["pitch_classes"], steps)
                transposed["measures"][i]["groups"][j] = group
        return transposed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=("charlie_parker", "bebop", "nottingham"), type=str, 
                        required=True, help="the dataset to convert")
    parser.add_argument("-o", "--output", default="ticks", choices=("ticks", "pitch_dur"),
                        help="The output format of the processed data.")
    parser.add_argument("-t", "--transpose", action="store_true",
                        help="Whether or not to transpose the parsed songs into all 12 keys.")
    args = parser.parse_args()

    if args.output == "ticks":
        p = TickParser(dataset=args.dataset)
    else:
        p = PitchDurParser(dataset=args.dataset)
    p.save_parsed(transpose=args.transpose)
