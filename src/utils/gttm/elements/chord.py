"""
A class that encapsulates multiple keys to be instantiated at the same point
in time.

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""

# package uk.ac.kent.computing.gttm.Elements;
# import java.util.*;
from key import Key

# public class Chord {
class Chord:
    # private List<Key> chord;

    # public Chord(List<Key> chordList) {
    def __init__(self, chordList):
        """
        :param chordList: (List<Key>) A list of Key objects to be included as part of the Chord object
        """
        # this.chord = chordList;
        self._chord = chordList
        # arrangeByLowestPitch();
        self._arrangeByLowestPitch()
    # }

    # public Chord() {
    @classmethod
    def initEmpty(cls):
        """
        Create an empty Chord object
        """
        # chord = new ArrayList<>();
        # arrangeByLowestPitch();
        return cls([])
    # }

    # public static Chord copyChord(Chord chrd) {
    @classmethod
    def copyChord(cls, chrd):
        """
        Creates a deep copy of a Chord object.
        :param chrd: (Chord) the Chord object to copy
        :return: (Chord) a copy of the Chord object
        """
        # Chord chrdCopy = new Chord();
        keyList = []
        # for (Key k : chrd.getKeys()) {
        for k in chrd.getKeys():
            # Key keyCopy = new Key(k.getNote(), k.getPosition());
            kCopy = Key.initCopy(k)
            # chrdCopy.addNote(keyCopy);
            keyList.append(kCopy)
        # }
        # return chrdCopy;
        return cls(keyList)
    # }

    # private void arrangeByLowestPitch() {
    def _arrangeByLowestPitch(self):
        """
        Arranges all the Keys that make up this Chord object such that
        they are in order of lowest pitch to highest pitch
        """
        # List<Key> retChord = new ArrayList<>();
        retChord = []
        # while (!chord.isEmpty()) {
        while len(self._chord):
            lowestPitch = self.__class__.findLowestPitch(self._chord)
            # retChord.add(findLowestPitch(chord));
            retChord.append(lowestPitch)
            # chord.remove(findLowestPitch(chord));
            self._chord.remove(lowestPitch)
        # }
        # chord = retChord;
        self._chord = retChord
    # }

    # private Key findLowestPitch(List<Key> chrd) {
    @staticmethod
    def findLowestPitch(chrd):
        """
        Identifies the lowest pitch of this Chord object
        :param chrd: (List<Key>) the list of Key objects from which to identify the lowest pitch
        :return: (Key) the lowest pitch
        """
        # List<Key> contenders = new ArrayList<>();
        contenders = []
        # KeyPositionEnum lowestPosition = chrd.get(0).getPosition();
        lowestPosition = chrd[0].getPosition()
        # for (Key k : chrd) {
        for k in chrd:
            # if (k.getPosition().getPosition() < lowestPosition.getPosition()) {
            if k.getPosition().getPosition() < lowestPosition.getPosition(): 
                # lowestPosition = k.getPosition();
                lowestPosition = k.getPosition()
                # contenders = new ArrayList<>();
                # contenders.add(k);
                contenders = [k]
            # } else {
            else:
                # if (k.getPosition() == lowestPosition) {
                if k.getPosition() == lowestPosition():
                    # contenders.add(k);
                    contenders.append(k)
                # }
            # }
        # }

        # Key lowestNote = contenders.get(0);
        lowestNote = contenders[0]
        # for (Key k : contenders) {
        for k in contenders:
            # if (k.getNote().getKeyNumber() < lowestNote.getNote().getKeyNumber()) {
            if k.getNote().getKeyNumber() < lowestNote.getNote().getKeyNumber():
                # lowestNote = k;
                lowestNote = k
            # }
        # }
        # return lowestNote;
        return lowestNote
    # }

    # public void addNote(Key key) {
    def addNote(self, key):
        """
        Add a new pitch to the collection of pitches contained within this Chord
        :param key: the pitch to add to the chord
        """
        # chord.add(key);
        self._chord.append(key)
        # arrangeByLowestPitch();
        self._arrangeByLowestPitch()
    # }

    # public List<Key> getKeys() {
    def getKeys(self):
        """
        :return: (List<Key>) all the pitches contained within this Chord object
        """
        # return chord;
        return self._chord
    # }
# }
