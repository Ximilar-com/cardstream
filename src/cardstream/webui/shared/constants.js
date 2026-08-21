// Values that must read the same on both sides of the wire.
//
// Mirrored from cardstream/core/id_types.py and pinned by a test there, so a
// change on either side fails the suite rather than silently making every
// "leave this out" dropdown send an unrecognised string.

// The "leave this field out" option in every optional select.
export const NOT_SPECIFIED = "Not Specified";
