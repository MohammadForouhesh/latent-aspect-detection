from termcolor import colored
from datetime import datetime
time_logging = lambda time_stamp: '[' + str(time_stamp.strftime("%b")) + '/' +\
                                        str(time_stamp.day)            + ' ' +\
                                        str(time_stamp.hour)           + ':' +\
                                        str(time_stamp.minute) + ']'

warn_logging = lambda text: text if len(text) >= 45 else '='*int((45-len(text))/2) + text + '='*int((45-len(text))/2)

compact_logging = lambda time_stamp, text, warning: [colored(time_logging(time_stamp), 'cyan'),
                                                     colored(warn_logging(text.upper().replace(' ', '=')), 'red'),
                                                     colored(warn_logging(warning))]

logger = lambda time_stamp, text, warning: '\n'.join(compact_logging(time_stamp, text, warning)[:3 if len(warning) != 0 else -1])

print_process_logger = lambda time_stamp, process_name, ratio: print(colored(time_logging(datetime.now()) +\
                                                                     '\t{:.2f}% of {} is completed.'.format(ratio*100, process_name), 'cyan')) \
                                                                if True in [(datetime.now() - time_stamp).seconds % number == 0
                                                                            for number in range(115, 120)] else ''

