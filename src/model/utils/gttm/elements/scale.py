"""
A class used to encapsulate a diatonic scale.

@author Alexander Dodd
"""
# package uk.ac.kent.computing.gttm.Elements;
# import java.util.*;
# import uk.ac.kent.computing.gttm.Manipulators.ScaleModeEnum;
import copy
import os.path as op
import sys
from pathlib import Path

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from manipulators import ScaleModeEnum, NoteOutOfBoundsException

# public class Scale{
class Scale:
   # private final ScaleModeEnum scaleEnum;
   # private final List<KeyLetterEnum> notes;
   # private KeyLetterEnum tonic;
    
    # public Scale(List<KeyLetterEnum> scaleNotes, ScaleModeEnum scaleName) {
    def __init__(self, scaleNotes, scaleName):
        """
        Create a Scale object from a given list of KeyLetterEnum and a given ScaleModeEnum
        :param scaleNotes: (List<KeyLetterEnum>) the list of keys to be included in the scale
        :param scaleName: (ScaleModeEnum) the scale mode of the Scale object
        """
        # notes = scaleNotes;
        self._notes = scaleNotes
        # tonic = notes.get(0);
        self._tonic = scaleNotes[0]
        # this.scaleEnum = scaleName;
        self._scaleEnum = scaleName # worrisome
    # }
  
    # public ArrayList<KeyLetterEnum> getNotes() {
    def getNotes(self):
        """
        :return: (ArrayList<KeyLetterEnum>) a list of all the KeyLetterEnum objects associated with the Scale
        """
        # ArrayList<KeyLetterEnum> copyList = new ArrayList<>();
        # self._notes.stream().forEach((l) -> {
            # copyList.add(l);
       # });
        # return copyList;
        return copy.deepcopy(self._notes)
    # }

    # public ScaleModeEnum getScaleModeEnum() {
    def getScaleModeEnum(self):
        """
        :return: (ScaleModeEnum) the scale mode of the Scale (e.g. Major or Minor)
        """
        return self._scaleEnum
    # }
    
    # public KeyLetterEnum getDominant() {
    def getDominant(self):
        """
        :return: (KeyLetterEnum) the dominant note of the scale
        """
        # return notes.get(4);
        return self._notes[4]
    # }
    
    # public KeyLetterEnum getTonic()
    def getTonic(self):
        """
        :return: (KeyLetterEnum) the tonic of the scale
        """
        return self._notes[0]
    # }
    
    # public KeyLetterEnum getSupertonic() {
    def getSupertonic(self):
        """
        :return: (KeyLetterEnum) the supertonic of the scale
        """
        # return notes.get(1);
        return self._notes[1]
    # }
    
    # public KeyLetterEnum getMediant() {
    def getMediant(self):
        """
        :return: (KeyLetterEnum) the mediant of the scale
        """
        # return notes.get(3);
        return self._notes[2]
    # }
    
    # public KeyLetterEnum getSubdominant() {
    def getSubdominant(self):
        """
        :return: (KeyLetterEnum) the subdominant of the scale
        """
        # return notes.get(3);
        return self._notes[3]
    # }
    
    # public KeyLetterEnum getSubmediant()
    def getSubmediant(self):
        """
        :return: (KeyLetterEnum) the submediant of the scale
        """
        # return notes.get(5);
        return self._notes[5]
    # }

    # public KeyLetterEnum getSubtonic()
    def getSubtonic(self):
        """
        :return: (KeyLetterEnum) the subtonic of the scale
        """
        # return notes.get(6);
        return self._notes[6]
    # }
# }
