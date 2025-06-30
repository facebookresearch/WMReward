#!/usr/bin/env python
#
# author: Mathieu Bernard <mathieu.a.bernard@inria.fr>
"""Make sure a zip archive is ready for submission"""

import argparse
import logging
import os
import sys
import zipfile


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
log = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'zipfile', help='The zip archive you are going to submit')
    parser.add_argument(
        'taskfile', help='The task file associated with the submission')
    return parser.parse_args()


# def get_block(taskfile):
#     if 'O1' in taskfile:
#         return 'O1'
#     elif 'O2' in taskfile:
#         return 'O2'
#     elif 'O3' in taskfile:
#         return 'O3'
#     else:
#         raise ValueError('{}: task unknown'.format(taskfile))



def main():
    args = parse_arguments()

    if not os.path.isfile(args.taskfile):
        raise ValueError('{}: task file not found'.format(args.taskfile))

    log.info('validating %s ...', args.zipfile)
    task = {line.strip() for line in open(args.taskfile, 'r')}

    log.info('check zip extension ...')
    if not args.zipfile.endswith('.zip'):
        raise ValueError(
            '{}: does not end with .zip extension'.format(args.zipfile))

    log.info('check valid zip format ...')
    if not zipfile.is_zipfile(args.zipfile):
        raise ValueError(
            '{}: not a valid zip file'.format(args.zipfile))

    with zipfile.ZipFile(args.zipfile) as myzip:
        content = myzip.namelist()

        log.info('check answer.txt is in zip ...')
        if not 'answer.txt' in content:
            raise ValueError(
                'answer.txt not present in zip, content is: {}'
                .format(', '.join(content)))

        log.info('check answer.txt is the only file in zip ...')
        if not content == ['answer.txt']:
            raise ValueError(
                'zip file must contains "answer.txt" only but contains: {}'
                .format(', '.join(content)))

        log.info('check structure of answer.txt ...')
        with myzip.open('answer.txt', 'r') as answerfile:
            entries, scores = [], []
            for i, line in enumerate(answerfile):
                line = line.decode('utf8')
                splitline = line.split()
                if len(splitline) != 2:
                    raise ValueError(
                        'lines must have 2 fields, line {} has {}'
                        .format(i+1, len(splitline)))

                entries.append(splitline[0])
                scores.append(float(splitline[1]))

            log.info(
                'check entries are valid for in answer.txt ...')
            entries = set(entries)
            diff = entries.symmetric_difference(task)
            if len(diff) != 0:
                raise ValueError(
                    'entries in answer.txt do not match entries in {}'
                    .format(args.taskfile))

            log.info('check answers are in [0, 1] in answer.txt ...')
            for i, s in enumerate(scores):
                if not 0 <= s <= 1:
                    raise ValueError(
                        'answers must bi in [0, 1], for line {} it is: {}'
                        .format(i+1, s))

    log.info('submission is valid, ready to be submitted!')


if __name__ == '__main__':
    try:
        main()
    except ValueError as err:
        log.error(err)
        log.error('invalid submission, correct errors and try again')
        sys.exit(1)
    else:
        sys.exit(0)
