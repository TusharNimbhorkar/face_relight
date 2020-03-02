# ==========================================
#  Project:  Common Tools
#  Author: Nedko Savov
#  Date:
# ==========================================


import re
import sys

import commons.globals as globals

def __tryint(s):
    try:
        return int(s)
    except:
        return s


def __alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [__tryint(c) for c in re.split('([0-9]+)', s)]


def sort_numerically(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=__alphanum_key)

class BColors:
    ResetAll = "\033[0m"

    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"

    BackgroundDefault = "\033[49m"
    BackgroundBlack = "\033[40m"
    BackgroundRed = "\033[41m"
    BackgroundGreen = "\033[42m"
    BackgroundYellow = "\033[43m"
    BackgroundBlue = "\033[44m"
    BackgroundMagenta = "\033[45m"
    BackgroundCyan = "\033[46m"
    BackgroundLightGray = "\033[47m"
    BackgroundDarkGray = "\033[100m"
    BackgroundLightRed = "\033[101m"
    BackgroundLightGreen = "\033[102m"
    BackgroundLightYellow = "\033[103m"
    BackgroundLightBlue = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan = "\033[106m"
    BackgroundWhite = "\033[107m"

def colored(text, color):
    return color + str(text) + BColors.ResetAll

class Logger:
    def __init__(self, tag, log_level=0, tag_color=BColors.Green):
        '''
        A logging class used to communicate information with the user
        :param tag: The string that identifies which component gives the message
        :param log_level: minimum priority level (0 to 5) to print
        '''
        self.tag = tag
        self.log_level=log_level
        self.tag_color = tag_color

        if tag == '':
            self.tag_text = ''
        else:
            self.tag_text = colored("[%s] " % (self.tag), self.tag_color)

    def __process_args(self, message, *args):
        message = str(message)
        for arg in args:
            if not(isinstance(arg, list) and len(arg)==0):
                message+=' ' + str(arg)

        return message

    def __print(self,message, color, ops):

        replace = False
        if 'replace' in ops.keys():
            replace=ops['replace']

        if replace:
            sys.stdout.write("\r" + self.tag_text + colored(message, color))
        else:
            print(self.tag_text + colored(message, color))

    def w(self, message, *args, **kwargs):
        '''
        Log warning
        :param message:
        :return:
        '''

        if self.log_level <=3:
            message = self.__process_args(message, *args)
            self.__print(message, BColors.Yellow, kwargs)

    def e(self, message, *args, **kwargs):
        '''
        Log error
        :param message:
        :return:
        '''

        if self.log_level <=5:
            message = self.__process_args(message, *args)
            self.__print(message, BColors.Red, kwargs)

    def i(self, message, *args, **kwargs):
        '''
        Log info
        :param message:
        :return:
        '''
        if self.log_level<=1:
            message = self.__process_args(message, *args)
            self.__print(message, BColors.LightBlue, kwargs)


    def v(self, message, *args, **kwargs):
        '''
        Log verbose
        :param message:
        :return:
        '''

        if self.log_level==0:
            message = self.__process_args(message, *args)
            self.__print(message, BColors.ResetAll, kwargs)

    def d(self, message, *args, **kwargs):
        '''
        Log debug
        :param message:
        :return:
        '''

        if globals.DEBUG:
            message = self.__process_args(message, *args)
            self.__print(message, BColors.LightMagenta, kwargs)

def_log = Logger('Log')