# Face Recognizer

Python script to recognize faces from video

## Requirements
- Install requirements using `pip install -r requirements.txt`

IMPORTANT NOTE: It's very likely that you will run into problems when pip tries to compile the dlib dependency. If that happens, check out this guide to installing dlib from source (instead of from pip) to fix the error:

[How to install dlib from source](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

After manually installing dlib, try running pip3 install face_recognition again to complete your installation.
## Usage
Run it using

```python main.py -i known_images -v video.mp4 -o output.csv --upsample-rate 2 ```

### Command line arguments
- **-i** or **--images-dir** Path to the directory containing training images
- **-v** or **--video** Path to the video
- **-o** or **--output-csv** The csv file to write the output.
- **-u** or **--upsample-rate** [Integer] How many times to upsample the image looking for faces. Higher numbers find smaller faces. 

### Recognize labels
Edit `label_pattern` variable in the file `main.py`
The sample pattern is 
```
{
    "shah": "Shahrukh Khan",
    "kapil": "Kapil Sharma"
}
```
For example, if the file names are like `shah01.jpg, shah02.jpg, shah03.jpg` for the label `Shahrukh Khan`. \
If the dict `key` is found in the image name then the label will be the `value`
For multiple persons add multiple key and values for each person.

## Output
The output csv will be in this format
``` 
8.84,   ['Kapil Sharma']
8.88,   ['Kapil Sharma']
8.92,   ['Kapil Sharma']
8.96,   ['Kapil Sharma']
9.0,    ['Kapil Sharma']
9.04,   ['Kapil Sharma']
9.08,   ['Kapil Sharma']
9.12,   ['Kapil Sharma']
9.16,   []

```
The Delimiter is `,`. The first column is `time` elapsed in video and the second column is `list of labels`.

## Credits

[face_recognition](https://github.com/ageitgey/face_recognition) library
