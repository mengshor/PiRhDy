from functions import collect_chords, merge_chords, get_chord_index
# collect chord

dirs=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']

for name in dirs:
    collect_chords(name)

# get chord dict and counts
merge_chords()

# get chord index
get_chord_index()