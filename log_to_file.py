import datetime
import colorama


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LogToFile:

    def __init__(self):
        colorama.init()
        now = datetime.datetime.now()
        self.filename = "Logs/log_" + str(now.day) + str(now.month) + str(now.year) + "_ " + str(now.hour) + str(now.minute) + ".txt"
        self.text = ""

    def log(self, t, p=True):
        self.text += t + "\n"
        if p:
            print(t)

    def print(self, t):
        print(bcolors.WARNING + t + bcolors.ENDC)

    def write(self):
        f = open(self.filename, "a")
        f.write(self.text + "\n\n")
        f.close()
        self.text = ""
