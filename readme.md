# CardioMusic

# Detail Result

## Music Inference

We predict ten music clips from test set using the music branch. The ground truth and predict label are plotted in the following figure.

![img_1.png](img_1.png)

we can see that the music brance of CardioMusic is effective.
## The Similarities of four ECG records and ten music clips. 

The Similarities of four ECG records and ten music clips are shown below, and the samples of ECG records and music clips can be find in [this direction](./samples). If the similarity is more than 0.5, then we can say the ECG and music are similar.

The ground truth label of the four ECG signals are (0.25, 0.25), (0.25,0.75), (0.75,0.75),(0.75, 0.25).

![img.png](img.png)

From this table, we can find that the ECG A is similar to music clip 9, ECG B and D is similar to nothing, and ECG C is most similar to music clip 7. 

![img](ECG-music.png)

We can conclude that the ECGs with low emotion VA are tend to be recommend the music with similar emotion which is gentle and sad, while the ECGs with high emotion VA are prefer the passionate and happy music. 
