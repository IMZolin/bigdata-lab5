# Big Data, lab 5

### System requirements:
- Python 3.10
- Java 17+ (for pyspark)

### Set up:

0. Java installation (for MacOS)

```bash
brew install openjdk@17

export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH

sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk \
    /Library/Java/JavaVirtualMachines/openjdk-17.jdk

/usr/libexec/java_home -V
```

Verify the installation:
```bash
java -version
```

1. Install the Project

Clone the repository and install dependencies:

```bash
git clone git@github.com:IMZolin/bigdata-lab5.git
```

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Pipeline

Preprocess data and train the model:

```bash
python -m src.preprocess
python -m src.train
```

Make predictions:

```bash
python -m src.predict
```