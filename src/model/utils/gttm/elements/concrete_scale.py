"""
A concrete Scale extends the Scale class to encapsulate a Scale made up of
specific pitches.

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""

# package uk.ac.kent.computing.gttm.Elements;
# import java.util.*;
# import uk.ac.kent.computing.gttm.Manipulators.NoteOutOfBoundsException;
# import uk.ac.kent.computing.gttm.Manipulators.ScaleModeEnum;

import os.path as op
import sys
from pathlib import Path

from scale import Scale
from key import Key
from enums import KeyPositionEnum

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from manipulators import ScaleModeEnum, NoteOutOfBoundsException

# public class ConcreteScale extends Scale{
class ConcreteScale(Scale):
    
  # private  List<Key> scaleKeys = new ArrayList<>();

    # public ConcreteScale(List<KeyLetterEnum> scaleNotes, ScaleModeEnum scaleName, KeyPositionEnum position) throws NoteOutOfBoundsException {
    def __init__(self, scaleNotes, scaleName, position): 
        """
        Creates a concrete scale object
        :param scaleNotes: (List<KeyLetterEnum>) all the keys of the scale
        :param scaleName: (ScaleModeEnum) the type of the scale
        :param position: (KeyPositionEnum) the position of the tonic pitch

        throws NoteOutOfBoundsException if the scale contains any pitches
        that are below the first position or above the eighth position.
        """
        # super(scaleNotes, scaleName);
        super().__init__(scaleNotes, scaleName);
        self._scaleKeys = []
        # KeyLetterEnum lastKey = null;
        lastKey = None 
        # for (KeyLetterEnum let : scaleNotes) {
        for let in scaleNotes:
            # if (lastKey == null) {
            if lastKey is None:
                # lastKey = let;
                lastKey = let
            # } else {
                # if (let.getKeyIdentity() < lastKey.getKeyIdentity()) {
            elif let.getKeyIdentity() < lastKey.getKeyIdentity():
                # position = KeyPositionEnum.getNextPosition(position);
                position = KeyPositionEnum.getNextPosition(position)
                # lastKey = let;
                lastKey = let
                # }
            # }
            # scaleKeys.add(new Key(let, position));
            self._scaleKeys.append(Key(let, position))
        # }
        # if(scaleNotes.get(0).getKeyIdentity() < lastKey.getKeyIdentity()) {
        if self._scaleNotes[0].getKeyIdentity() < lastKey.getKeyIdentity():
            # KeyPositionEnum.getNextPosition(position);
            KeyPositionEnum.getNextPosition(position)
        # }
        # scaleKeys.add(new Key(scaleNotes.get(0), position));
        self._scaleKeys.append(Key(scaleNotes[0], position))
    # }
    
    # public List<Key> getScaleKeys() {
    def getScaleKeys(self):
        """
        :return: (List<Key>) a list of all the pitches contained within the scale
        """
        # return scaleKeys;
        return self._scaleKeys
    # }
# }
