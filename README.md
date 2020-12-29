# PiRhDy
code for our ACM MM 2020 best paper "PiRhDy: Learning Pitch-, Rhythm-, and Dynamics-aware Embeddings for Symbolic Music"

step 1: normalize original midi files: time normalization, key tranformation, etc.


step 2: transform midi files into time-pitch matrices


step 3: analysis chord in midi file: not necessary to re-run the files, all needed files already in this dir


step 4: transform matrices into quadruple sequences: (chroma, octave, velocity, state), the final format


step 5: 


        1) generate datasets for token modeling dataset 
        
        2) token modeling, containing pre-trained models


step 6: 


        1) transform sequence to bars 
        
        
        2) transform bars into phrases
        
        
        3) generate dataset for context modeling 
        
        
        4) context modeling and downstream tasks, embeddings pre-trained through token modeling is in "embeddings", models fine-tuned by context modeling is in "pre-trained models".
        



