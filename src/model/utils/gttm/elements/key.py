"""
A Key class is used to encapsulates details concerning a single pitch event.

@author Benjie Genchel, adapted from code by Alexander Dodd
ref: https://github.com/alexanderbdodd/GTTM_Music_Generator
"""
# import java.util.Comparator;
from enums import KeyLetterEnum as kle, KeyPositionEnum as kpe

# public class Key implements Comparator {
class Key:
    # private final KeyLetterEnum key;
    # private KeyPositionEnum position;

    # public Key(KeyLetterEnum key, KeyPositionEnum position) {
    def __init__(self, key, position):
        """
        Create a Key object with the given KeyLetterEnum and KeyPositionEnum
        :param key: (KeyLetterEnum) the key position of the pitch
        :param position: (KeyPositionEnum) the position of the pitch
        """
        self._key = key
        self._position = position

    # public Key(Key key) {
    @classmethod
    def initCopy(cls, toCopy):
        """
        Copies a Key object using the given Key object. Copy Constructor.
        :param key: (Key) the Key object to copy
        """ 
        return cls(toCopy.getNote(), toCopy.getPosition())

    # public KeyLetterEnum getNote() {
    def getNote(self):
        """
        :return: (KeyLetterEnum) the KeyLetterEnum associated with the Key object
        """
        return self._key;

    # public KeyPositionEnum getPosition() {
    def getPosition(self):
        """
        :return (KeyPositionEnum) the KeyPositionEnum associated with this Key object
        """
        return self._position

    # @Override
    # public int hashCode() {

    #     int hashcode = 0;
    #     hashcode = key.getKeyIdentity();
    #     hashcode += position.getPosition() * 100;

    #     return hashcode;
    # }
    def hashCode(self):
        hashcode = 0
        hashcode = key.getKeyIdentity()
        hashcode += self._position.getPosition() * 100
        return hashcode

    # @Override
    # public boolean equals(Object obj) {
        # if (obj.hashCode() == hashCode()) {
            # return true;
        # } else {
            # return false;
        # }
    # }
    def __eq__(self, obj):
        return obj.hashCode() == this.hashCode()


    # public int compare(Object obj1, Object obj2) {
    def compare(self, obj1, obj2):
        """
        Provides comparison of two Key objects
        :param obj1: (Key) first Key object to compare
        :param obj2: (Key) second Key object to compare
        :return: -1 if obj1 is in a lower position than obj2, 0 if they are in the
        same position, and 1 if obj1 is in a greater position than obj2
        """
        # Key o1 = (Key) obj1;
        # Key o2 = (Key) obj2;
        # if (o1.getNote() == KeyLetterEnum.BS && o1.getPosition() == KeyPositionEnum.getLastPosition(o2.getPosition())
                # && o2.getNote() == KeyLetterEnum.C) {
            # return 0;
        # }
        if (obj1.getNote() == kle.BS) and (obj1.getPosition() == kpe.getLastPosition(obj2.getPosition())) and (
                obj2.getNote() == kle.C):
            return 0

        # if (o1.getNote() == KeyLetterEnum.C
        #         && o2.getPosition() == KeyPositionEnum.getNextPosition(o1.getPosition())
        #         && o2.getNote() == KeyLetterEnum.B) {
        #     return 0;
        # }
        if (obj1.getNote() == kle.C) and (obj2.getPosition() == kpe.getNextPosition(obj1.getPosition())) and (
            obj2.getNote() == kle.B):
            return 0

        # if (o1.getPosition().getPosition() > o2.getPosition().getPosition()) {
            # return 1;
        # }
        # if (o1.getPosition().getPosition() < o2.getPosition().getPosition()) {
            # return -1;
        # } else {
            # if (o1.getNote().getKeyNumber() == o2.getNote().getKeyNumber()) {
                # return 0;
            # } else if (o1.getNote().getKeyIdentity() < o2.getNote().getKeyIdentity()) {
                # return -1;
            # } else {
                # return 1;
            # }
        # }
        if (obj1.getPosition().getPosition() > obj2.getPosition().getPosition()):
            return 1
        elif obj2.getPosition().getPosition() < obj2.getPosition().getPosition():
            return -1
        else:
            if obj1.getNote().getKeyNumber() == obj2.getNote().getKeyNumber():
                return 0
            elif obj1.getNote().getKeyIdentity() < obj2.getNote().getKeyIdentity():
                return -1
            else:
                return 1

    # public static Key getNextKey(Key lastKey) {
    @classmethod
    def getNextKey(cls, lastKey):
        """
        Returns the next pitch position above the given Key object
        :param lastKey: (Key) the Key from which to locate the next Key position
        :return: (Key) the next pitch position after the given Key object
        """
        # if (lastKey == null) {
        if lastKey is None:
            # return new Key(KeyLetterEnum.C, KeyPositionEnum.FIRST);
            return cls(kle.C, kpe.FIRST)
        # } else {
        else:
            # KeyLetterEnum letter = KeyLetterEnum.getNextNote(lastKey.getNote())[0];
            letter = kle.getNextNote(lastKey.getNote())[0]
            # KeyPositionEnum position;
            # if (letter.getKeyNumber() == KeyLetterEnum.C.getKeyNumber()) {
            if letter.getKeyNumber() == kle.C.getKeyNumber():
                # if (KeyPositionEnum.getNextPosition(lastKey.getPosition()) != null) {
                if kpe.getNextPosition(lastKey.getPosition()) is not None:
                    # position = KeyPositionEnum.getNextPosition(lastKey.getPosition());
                    position = kpe.getNextPosition(lastKey.getPosition())
                # } else {
                else:
                    # return null;
                    return None
                # }
            # } else {
            else:
                # position = lastKey.getPosition();
                position = lastKey.getPosition()
            # }
            # return new Key(letter, position);
            return cls(letter, position)
        # }
    # }

    # public void setPosition(KeyPositionEnum position) {
    def setPosition(self, position):
        """
        Sets a new position for the Key object
        :param position: (KeyPositionEnum) the new position of the Key object
        """  
        # this.position = position;
        self._position = position
    # }
# }
