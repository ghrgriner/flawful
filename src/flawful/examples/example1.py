#    Example 1 (German) language-learning flashcards
#    Copyright (C) 2024-2025 Ray Griner (rgriner_fwd@outlook.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------

"""Example 1 (German) language-learning flashcards"""

#------------------------------------------------------------------------------
# File:    example1.py (see previous names in change history)
# Date:    2024-12-31
# Author:  Ray Griner
#
# Purpose: Read in language database from tab-delimited text file, and select
#   fields and merge audio information for use in Anki (flashcard program).
#   There are various debug options for checking audio files without text
#   and vice-versa as well as functionality to compare with external text
#   word lists. See functionality section below for complete details.
#
# Output files:
#   output_notes.txt: One record per note, suitable for loading into Anki
#   output_notes_fields.txt: A single record with the field names for the
#     output_notes.txt file
#   wordlists_headwords_not_in_notes.txt: Listing of words from the input
#     reference lists that were not found in the database, or that did not have
#     a chapter assigned.
#   words_no_audio.txt: Listing of words from the input notes file that don't
#     have audio assigned.
#   de2_problems.txt: Listing of problems identified in the `de2` field, which
#     should in some circumstances have a certain number of tokens.
#
# Audio files copied and renamed
#
# Changes:
# [20250107] Change `assign_chapter` element for `de_xref` to True, and add
#   new field `de_xref_ignore_ch` where assign_chapter element is False. This
#   aligns more with expected use that there might be situations when the user
#   wants to assign the chapter and others when they do not.
# [20250111] Remove `de_en_add` references, and replace with references to
#   new fields `de1_hint`, `de_notes`, and `de3_prompt`.
# [20250112] Add call to `make_prompt_and_answer_table` and output the created
#   fields (`de_table_answer`,`de_table_prompt`,`de3_omitted`). Refactor code
#   per some linter warnings.
# [20250121] Add `import csv` and change `quoting=3` to quoting=csv.QUOTE_NONE
#   throughout. Add commented-out example that adds metadata to the output file
#   header.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# This is an example using about 40 input notes for German flashcards with
#  about ~10 fields in the input file. The `comments` field of the
#  input_notes.txt file gives the rationale / points of emphasis for each
#  record.
#
# The example illustrates the functionality described in the package docstring.
# Please refer to that docstring for details. The wiki on the github repository
# also gives an overview of the fields in the example input file, as well as
# recommendations that apply to production.
#------------------------------------------------------------------------------

import re
import os
import csv
from collections import namedtuple

import pandas as pd
import numpy as np
import flawful
import flawful.german
import flawful.add

#------------------------------------------------------------------------------
# Input / Output directories should be set by the user before running.
# (Getting the directory names using os.environ can be replaced by just
# setting the path with a string, as in the commented-out lines below.)
#
# The input test data is distributed with the package in the data/ directory
# and subdirectories.
#------------------------------------------------------------------------------
#INPUT_DIR = '/path/to/data'
INPUT_DIR = os.environ.get('FLAWFUL_EXAMPLE1_DATA')
#OUTPUT_DIR = '/path/to/output'
OUTPUT_DIR = os.environ.get('FLAWFUL_EXAMPLE1_OUTPUT')
#AUDIO_OUTPUT_DIR = '/path/to/output'
AUDIO_OUTPUT_DIR = os.environ.get('FLAWFUL_EXAMPLE1_AUDIO_OUTPUT')

#------------------------------------------------------------------------------
# The following constants can be modified by the user, but the defaults can be
# used for the example unmodified.
#------------------------------------------------------------------------------

PRINT_DUPLICATE_AUDIO_HEADWORDS = True
# If True, then all records in input file written to output in the same order
EXPORT_ALL_RECORDS = False
WRITE_WORDS_WITHOUT_AUDIO = True
PRINT_NOTES_WITHOUT_AUDIO = False
# Set to an integer to restrict some printing to just one chapter.
ONE_CHAPTER = None
# List of reference lists to compare, use 'All' for all
WORDLISTS_TO_COMPARE = 'All'

# Copy the used audio files to another directory. The output filename will be
#  set to f'{basename_out}.{ext}', where `ext` is the extension parameter that
#  was passed to `add_from_dir` when loading the audio files and
#  `basename_out` is the attribute returned by the `make_name_info` function
#  parameter of `add_from_dir`.
COPY_AUDIO = True
PRINT_UNUSED_AUDIO = True
# two output files generated, f'{prefix}.txt' and f'{prefix}_fields.txt'
OUTPUT_FILE_PREFIX = 'output_notes'
# also two output files generated, f'{prefix}.txt' and f'{prefix}_fields.txt'
DE_ADD_OUTPUT_FILE_PREFIX = 'de_additional'
# Set the below to `None` if the file is not being used.
#DE_ADDITIONAL_INPUT_FILE = None
DE_ADDITIONAL_INPUT_FILE = 'additional_input_notes.txt'

# If not `None`, `pd.DataFrame.rename` will be called on the dataset that makes
# the additional output file just before the output is written, and this will
# be passed as the mapper. This is because the default names generated have
# `en` prefixes (meaning English) and `de` prefixes (meaning German). Users
# can then store words in other languages in these fields and rename the fields
# to something appropriate. Users setting this value may also want to pass a
# different value than the defaults of `en_hint`, `de_hint`, and `htag_prefix`
# to `create_tl_additional_output` since the defaults are 'E', 'D', and 'DE',
# meaning English, German, and German, respectively.
#DE_ADD_OUTPUT_MAPPER = {'en1': 'po1'}
DE_ADD_OUTPUT_MAPPER = {'nl1': 'en1', 'tl2': 'de2', 'tl1': 'de1',
     'tl1_color': 'de1_color',
     'tl_pronun': 'de_pronun',
     'tl_defs': 'de_defs',
     'tl_table_prompt': 'de_table_prompt',
     'tl_table_answer': 'de_table_answer',
     'tl_audio':  'de_audio',
                       }

DE_ADD_INPUT_MAPPER = {'en1': 'nl1',
     'de1_hint': 'tl1_hint',     'de1_list': 'tl1_list',
     'de2_list': 'tl2_list',     'de_notes_list': 'tl_notes_list',
     'de_answer': 'tl_answer',   'de_answer_list': 'tl_answer_list',
     'en_answer': 'nl_answer',   'en_answer_list': 'nl_answer_list',
     'de_pronun': 'tl_pronun',   'de_headword': 'tl_headword',
     'de1': 'tl1',              'de2' : 'tl2',
     'de3p_1': 'tl3p_1', 'de3p_2': 'tl3p_2', 'de3p_3': 'tl3p_3',
     'de3p_4': 'tl3p_4', 'de3p_5': 'tl3p_5',
     'de3d_1': 'tl3t_1', 'de3d_2': 'tl3t_2', 'de3d_3': 'tl3t_3',
     'de3d_4': 'tl3t_4', 'de3d_5': 'tl3t_5',
     'de3e_1': 'tl3n_1', 'de3e_2': 'tl3n_2', 'de3e_3': 'tl3n_3',
     'de3e_4': 'tl3n_4', 'de3e_5': 'tl3n_5',
                       }
ADD_INPUT_MAPPER = DE_ADD_INPUT_MAPPER

# If not none, then if `create_tl_additional_output` is called, this value is
# passed in the `braces_html_class` parameter, so that text surrounded by
# braces in `de` or `de3_prompts` (or de3d_N, de3e_N, de3p_N) is put in an
# HTML div element with the indicated class.
BRACES_HTML_CLASS = 'highlight'

# See docstring in examples/de_additional.py for explanation of flags. In our
# (non-shared) code, we use '°' to refer to tokens that we might want a
# flashcard for and '†' for tokens that we don't (e.g., if we know a phrase
# on DE3 is already entered on another card). In the example input files for
# this public program, we only use '°' as flags, but there is no harm in
# specifying '†' also.
FLAGS = '°†'
# Can differ from the above. This is only used when checking
# `de3` vs `de3_prompt` to verify that when a token in one is flagged, so is
# the corresponding token in the other.
DE3_FLAGS_TO_CHECK = '°'

MAX_CHAPTER = 20

# Regular expression for strings where we don't want to remove the initial
# definite article(s) in the token when making the lookup key / headword
KEEP_INITIAL_DER_DAS_DIE_PATTERN = re.compile(
        '^die Niederlande|^der Esel sagt ihm')

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def tokenize_de2(de1, de2, part_of_speech):
    if part_of_speech == 'V' or (part_of_speech == 'N' and '(in)' in de1):
        return de2.split(';')
    else:
        return de2.split(',')

def select_output_columns(df_):
    """Select output columns from the data frame.

    It's sometimes useful to pass in the `EXPORT_ALL_RECORDS` variable
    as a parameter, so we have made this a function instead of a line
    of code.
    """
    return df_[['note_id','en1','part_of_speech','de_target_number',
           'de1','de1_at1_sd1_color','de2','de3','de3_color','de_xref_color',
           'de_xref_ignore_ch_color','de1_hint','de_table_prompt',
           'de_table_answer','de3_omitted',
           'de_rev_table','de_rev_table_prompt','de3_rev_omitted',
           'de_notes','de3_prompt',
           'de_conf','de_pronun','dib_sentences','at1','sd1','de_audio',
           'de_no_audio','chapter','de_sentences','tags']]

def make_headword_reflist(line):
    """Simple function to extract key (=headword) when making dictionaries.

    The key (headword) is obtained from taking the portion of the input
    before the first ',' or '(', and removing any definite articles at
    the beginning of the string, unless we have indicated we do not want
    to remove the article by including the beginning of the string in
    KEEP_INITIAL_DER_DAS_DIE_PATTERN.

    The dictionaries than use this function represent reference lists or
    okay-lists. In this example, the same function can be used to make
    the six dictionaries that hold the three reference lists and their
    associated three okay-lists, but in practice, a different key
    function might be needed for different reference lists, in which
    case the function can simply be defined in the appropriate
    `make_..._dict` function.
    """
    key = line.split(',')[0].split('(')[0]
    # we use the same translation table as `make_headword_notes`, but
    # note converting ',' to '_' will never happen since we just used
    # ',' to split the line.
    de_trtab = str.maketrans("'," ,'__' ,'?!°')
    if not KEEP_INITIAL_DER_DAS_DIE_PATTERN.match(key):
        key = flawful.german.INITIAL_DER_DAS_DIE_PATTERN.sub('', key)
    key = key.translate(de_trtab)
    return key.strip()

def make_wordlist_key_notes(x):
    """Extract key for lookup in Wordlists from string

    In this example, it will be called on each token in the notes file,
    where some fields may be semi-colon delimited and therefore the
    token can contain commas that should not be used as a delimiter. For
    this reason, it needs to be a little different from
    `make_headword_reflist`, which is meant to operate on comma-
    delimited lines.

    Other differences are to take the portion of the string before the
    first '[' (e.g., to handle tokens like 'gegen [acc.]'). Users may
    find it necessary to do the same in `make_headword_reflist`.

    However, it is sometimes convenient to have tokens in the input
    notes file like 'sich [um A] Sorgen machen' (='worry [about
    something']), and this would yield a headword of 'sich', causing the
    note to be incorrectly matched to reference list(s) with this
    headword. Therefore, 'sich' is only returned as a headword when the
    input token before parsing equals 'sich'. In other cases where the
    parsed headword is 'sich', '' is returned, and the expectation is
    the user can put the desired headword in 'de_xref' for linking to
    reference list(s).
    """
    ret_val = x.strip()
    de_trtab = str.maketrans("'," ,'__' ,'?!°')
    if not KEEP_INITIAL_DER_DAS_DIE_PATTERN.match(ret_val):
        ret_val = flawful.german.INITIAL_DER_DAS_DIE_PATTERN.sub('',
                                 ret_val)
    ret_val = ret_val.translate(de_trtab)
    ret_val = ret_val.split('[')[0].split('(')[0]
    if ret_val.strip() == 'sich' and x.strip() != 'sich':
        ret_val = ''
    return ret_val.strip()

def make_audio_key_notes(x):
    """Extract key for lookup in audio file dictionaries from string.

    Like `make_wordlist_key_notes`, but '.' is removed from the string
    and '/' and '…' are replaced with '_'. This is because when we made
    audio files for the example sentences in Deutsch im Blick, this was
    the convention we adopted to omit these special characters from the
    file name, and the file name is then used to generate the key in the
    dictionary.

    Perhaps it would have been cleaner to simply put this same logic in
    `make_wordlist_key_notes` and `make_headword_reflist` as well, but
    we have not done so.
    """
    ret_str = make_wordlist_key_notes(x).replace('/','_')
    return ret_str.replace('.','').replace('…','_')

def make_la_dict():
    """Make the dictionary (`data` attribute) for reference list 'LA'.

    In this example, the input file is formatted so that the lines are
    grouped by chapter, where a line of '^Chapter n' starts the section
    for chapter n.

    Returns
    -------
    Dictionary where the key is the headword and value is a
    WordlistEntry object.
    """
    ignore_line_pattern = re.compile('^#')

    chapter_num = 0
    ret_dict = {}
    with open(os.path.join(INPUT_DIR, 'reflist_LA.txt'),
              encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('^Chapter '):
                chapter_num = int(line.replace('^Chapter ',''))
                continue
            # ignore blank lines and comments
            if (not line) or ignore_line_pattern.match(line):
                continue
            dictkey=make_headword_reflist(line)
            if dictkey in ret_dict:
                # can be changed to just print a message if needed
                raise ValueError(f'Duplicate key in reflist LA: {dictkey}')
            ret_dict[dictkey] = flawful.WordlistEntry(full_line=line,
                                              book_chapter=chapter_num)
    return ret_dict

def make_lb_dict():
    """Make the dictionary (`data` attribute) for reference list 'LB'.

    In this example, the input file is formatted so that the input is
    tab-delimited, where the first column is German text that must be
    parsed to extract the headword, and the second column contains the
    chapter formatted as 'n/A', where n is the chapter number and 'A' is
    an arbitrary letter, perhaps representing the section within the
    chapter.

    Returns
    -------
    Dictionary where the key is the headword and value is a
    WordlistEntry object.
    """
    df_ = pd.read_csv(os.path.join(INPUT_DIR, 'reflist_LB.txt'),
        comment='#', sep='\t', na_filter=False, quoting=csv.QUOTE_NONE,
        names=['Full Word', 'Chapter/Section'])

    df_['key'] = df_['Full Word'].map(make_headword_reflist)
    df_['chapter'] = df_['Chapter/Section'].map(lambda x: int(x.split('/')[0]))
    flawful.dupkey(df_, ['key'], additional_vars=['Full Word'],
                   desc='reflist_lb', ifdup='print')
    ret_dict = {}
    for row in df_[['key','Full Word','chapter']].values:
        ret_dict[row[0]] = flawful.WordlistEntry(full_line=row[1],
                                     book_chapter=row[2])
    return ret_dict

def make_lc_dict():
    """Make the dictionary (`data` attribute) for reference list 'LC'.

    In this example, the input file is formatted so that the input has
    columns 'Base Word', 'Full Word', 'Chapter', 'S1', 'S2', and 'S3',
    where the last three columns are example sentences that will be put
    in the `examples` attribute of the `WordlistEntry` objects stored in
    the dictionary.

    Returns
    -------
    Dictionary where the key is the headword and value is a
    WordlistEntry object.
    """
    df_ = pd.read_csv(os.path.join(INPUT_DIR, 'reflist_LC.txt'),
        comment='#', sep='\t', na_filter=False, quoting=csv.QUOTE_NONE)

    df_['key'] = df_['Base Word'].map(make_headword_reflist)
    flawful.dupkey(df_, ['key'], additional_vars=['Base Word'],
                   desc='reflist_lc', ifdup='print')
    ret_dict = {}
    for row in df_[['key','Full Word','Chapter','S1','S2','S3']].values:
        examples = [x for x in row[3:] if x != '']
        if row[2]:
            ret_dict[row[0]] = flawful.WordlistEntry(full_line=row[1],
                   book_chapter=int(row[2]), examples=examples)
        else:
            ret_dict[row[0]] = flawful.WordlistEntry(full_line=row[1],
                   examples=examples)
    return ret_dict

def select_output_rows(df_, max_chapter, one_chapter, export_all_records):
    """Simple function to subset input data frame.
    """
    if export_all_records:
        return df_

    if not one_chapter:
        ret_df = df_[df.chapter <= max_chapter]
    else:
        ret_df = df_[df.chapter == one_chapter]

    return ret_df

def preprocess_de_conf(de_conf):
    """Expand abbreviations in input string.

    Only one abbreviation is given as an example, the string 'VBZIEHEN'
    is expanded to a string with all the German words (in the example)
    with 'ziehen' as the base, along with their English definitions.
    """

    ret_val = de_conf
    ret_val = ret_val.replace('VBZIEHEN','[sich anziehen (get dressed), sich ausziehen (get undressed), sich umziehen (get changed), alleinerziehende Mutter (single mother), Kinder erziehen (raise children)]') # pylint: disable=line-too-long
    # ret_val = ret_val.replace(..., ...)
    return ret_val

def make_okay_dict(input_file, desc):
    """Make an okay-list from a tab-delimited input file.

    Only the first column of the input will be used. This column will be
    parsed to get the headword.

    Returns
    -------
    Dictionary where the key is the headword and value is a
    OkaylistEntry object.
    """
    int_df = pd.read_csv(input_file, comment='#', sep='\t',
                         na_filter=False, quoting=csv.QUOTE_NONE,
                         names=['Word'])
    flawful.dupkey(int_df, ['Word'], desc=desc, ifdup='print')

    int_df['dictkey'] = int_df['Word'].map(make_headword_reflist)
    flawful.dupkey(int_df, ['dictkey'], desc=desc, additional_vars=['Word'],
                   ifdup='print')

    ret_dict = {}
    for row in int_df[['dictkey','Word']].values:
        ret_dict[row[0]] = flawful.OkaylistEntry(row[1])
    return ret_dict

def str_to_chapter(x):
    """Convert chapter formatted as a string to a number.

    This is used in cases where the user has indicated in the input
    notes file the chapter(s) the words in a note are from. It's
    probably best to assign chapters in the `Wordlist` object, but
    there may be cases where the user wanted to put the value in the
    input notes file.

    The chapter might start with different user-defined letter to which
    some value can be added so that chapters from different books are
    assigned non-overlapping numeric chapters that define the order for
    study. In this example, we assume that the input is either a float
    that can be converted to an integer 1-10, or a string 'Fn', where
    the chapter will be (10+n).

    In this example code, this is used to get a chapter from the input
    notes file, but the final chapter assigned to a note may be lower if
    a lower chapter is found via a reference list.
    """
    if not x:
        return flawful.DEFAULT_CHAPTER_FROM_NOTES

    if x.strip().startswith('F'):
        return 10 + int(x.strip()[1:2])

    return int(np.floor(float(x)))

def filter_text_not_audio_pre(chapter, in_wordlists, token):
    """Indicate whether to put the token in the 'keys_no_audio' dict.

    Function is passed to `tag_audio_and_markup` in this example.
    """
    return ((in_wordlists['LA'] or in_wordlists['LB'] or in_wordlists['LC'])
                and not flawful.german.VOWEL_LENGTH_PATTERN.search(token))

def filter_text_not_audio_post(x):
    """Indicate whether to print the token in the 'keys_no_audio' dict.

    Function is passed to `german.write_words_no_audio` in this example
    to filter at the time of writing, but we probably could have just
    combined the logic here with that in `filter_text_not_audio` above.
    """
    return (' ' not in x and '~' not in x and '_' not in x and
            '=' not in x and '≈' not in x)

def make_name_info(stem):
    """Convert input file stem to output file stem (=basename) and headword.

    Assumes the headword can be obtained from starting at position 8 of
    the input stem. This is true for the files in the data/audio/DiB
    directory provided for this example, as these files start with
    'DiB-Dnn_', where nn is the chapter in Deutsch im Blick.
    """
    headword = stem[8:]
    base_out = f'flawful_ex1_{headword}'
    return flawful.AudioFileNameInfo(headword=headword, basename_out=base_out)

def make_name_info_numbered(stem):
    """Convert input file stem to output file stem (=basename) and headword.

    Assumes the headword can be obtained from starting at position 3 of
    the input stem, and then converting '_' to ' ' and removing digits.
    This is true for the files in the 'data/audio/numbered_ogg'
    directory provided for this example, as these files start with
    'De-'. This is the most common naming convention for German audio
    files in the German Wiktionary (de.wiktionary.org), although the
    files provided with this package are renamed Deutsch im Blick files
    and not taken from the Wiktionary.
    """
    headword = stem[3:].translate(str.maketrans('_',' ','0123456789'))
    base_out = f'flawful_ex1_{headword}'
    return flawful.AudioFileNameInfo(headword=headword, basename_out=base_out)

def load_audios_to_dict(input_dict, print_duplicate_headword):
    """Load audio file information to custom dictionary object.

    The files are taken from two example directories, one where the
    files are formatted as in the unofficial Deutsch im Blick Anki deck,
    and the second where the most common naming convention for German '
    Wiktionary files are used.

    Parameters
    ----------
    input_dict : `AudioFileDict` object
        Dictionary that will have values added.
    print_duplicate_headword : bool
        Prints to stdout when a headword already exists in `input_dict`.

    Returns
    -------
    None, but side-effect is that the input dictionary has items added
    for each new headword derived from an audio file.
    """
    input_dict.add_from_dir(ext='mp3',
        dirpath=os.path.join(INPUT_DIR, 'audio', 'DiB'),
        make_name_info=make_name_info,
        print_duplicate_headword=print_duplicate_headword)

    input_dict.add_from_dir(ext='ogg',
        dirpath=os.path.join(INPUT_DIR, 'audio', 'numbered_ogg'),
        make_name_info=make_name_info_numbered,
        print_duplicate_headword=print_duplicate_headword)

def make_known_no_audio_dict():
    """Make dictionary containing audio lookup keys known to have no audio.

    Assumes tab-delimited input file has columns 'Word' and 'Reason',
    then simply puts this into a dictionary where 'Word' is the key
    and 'Reason' the value.
    """
    df_ = pd.read_csv(os.path.join(INPUT_DIR, 'known_no_audio.txt'),
            comment='#', sep='\t', skiprows=(0), na_filter=False,
            quoting=csv.QUOTE_NONE)
    flawful.dupkey(df_, ['Word'], desc='known_no_audio.txt')
    #df['Reason'] = 'not found'
    df_.set_index('Word', inplace=True)
    return df_['Reason'].to_dict()

def make_tables_and_listings(df_: pd.DataFrame, wordlist_id,
                             print_notes_without_audio, one_chapter):
    """Example function that writes some frequency tables and listings.

    Print some simple listings and descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Should have fields note_id, en1, de1, de3, chapter, de_audio.
    wordlist_id : str
        Id of word list
    print_notes_without_audio : bool (default = True)
        Print records where no audio found (de_audio == '')
    one_chapter : int, optional (default = None)
        If set, the listing generated when print_text_without_audio is
        True will be printed only for df.chapter == one_chapter. Otherwise,
        all records will be listed.

    Returns
    -------
    None, only side-effect is printing.
    """
    # Probably sensible to print a warning if the foreign word is missing.
    dfprint = df_[['note_id','en1','de1']]
    if len(dfprint[(df_.de1=='')])>0:
        print('\nWARNING: Records with de1 missing:')
        print(dfprint[(df_.de1=='')])

    #--------------------------------------------------------------------------
    # Cross tabs of Chapter by Audio file status, etc...
    #--------------------------------------------------------------------------
    flawful.twowaytbl(df_, title='\nNotes with German audio by chapter:',
              row='chapter', col='has_german_audio')
    if wordlist_id and wordlist_id != 'All':
        flawful.twowaytbl(df_, title=('\nNotes by German chapter and status in'
                  f' {wordlist_id} word list:'),
                  row='chapter', col=f'In{wordlist_id}', cumulative=True)

    #--------------------------------------------------------------------------
    # Print records from each unit with no German audio for any field.
    #--------------------------------------------------------------------------
    if print_notes_without_audio:
        if not one_chapter:
            dfprob = df_[ (df_.de_audio == '') ]
            dfprob = dfprob[['note_id','en1','de1','de3']]
            num_miss = len(dfprob.index)
            print(f'\nMissing German audio ({num_miss} records):')
        else:
            dfprob = df_[ (df_.chapter == one_chapter)
                            & (df_.de_audio == '') ]
            dfprob = dfprob[['note_id','en1','de1','de3']]
            num_miss = len(dfprob.index)
            print(f'\nMissing German audio ({num_miss} records):')
            if num_miss>0:
                print(dfprob)

    # other examples of descriptive tables
    #df_['FirstLB'] = (df_['InLB'] & ~df_['InLA'])
    #print(df_.pivot_table(index=['FirstLB','InLB','InLA'], values='de1',
    #      aggfunc=len, margins=True))

def write_de2_problems(df_, outfile):
    """Example function that writes records flagged with problems in `de2`.
    """
    df_out = df_[['chapter','de1','de2','de2_problems']]
    df_out = df_out[~(dfout.de2_problems == '')]
    df_out.to_csv(outfile, sep='\t', index=False, quoting=csv.QUOTE_NONE)

#------------------------------------------------------------------------------
# End functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main entry point
#------------------------------------------------------------------------------

# Dictionary for storing full path to each audio file, as well as the key
audio_file_dict = flawful.AudioFileDict()

# Load audio files to the dictionary
load_audios_to_dict(audio_file_dict, PRINT_DUPLICATE_AUDIO_HEADWORDS)
known_no_audio = make_known_no_audio_dict()

# Load reference lists into dictionaries
# reference lists added first take priority if a headword is in multiple lists
# (for example, when creating HTML tags for color highlighting)
de_dicts = flawful.Wordlists()
de_dicts.add(list_id='LA', chapter_offset=0, data=make_la_dict())
de_dicts.add(list_id='LB', chapter_offset=3, data=make_lb_dict())
de_dicts.add(list_id='LC', chapter_offset=5, data=make_lc_dict())

aud_dicts = {'file_info': audio_file_dict,
             'keys_no_audio': {},
             'known_no_audio': known_no_audio}

# Load okay-lists into dictionaries
de_okay_dicts = {}
de_okay_dicts['LA'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LA.txt'), 'LA')
de_okay_dicts['LB'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LB.txt'), 'LB')
de_okay_dicts['LC'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LC.txt'), 'LC')

#------------------------------------------------------------------------------
# Read input file that is one record per note for the flashcards.
#------------------------------------------------------------------------------
df = pd.read_csv(os.path.join(INPUT_DIR, 'input_notes.txt'), sep='\t',
                 skiprows=(0), na_filter=False, quoting=csv.QUOTE_NONE)

flawful.dupkey(df, by_vars=['en1','part_of_speech'], desc='df',
               additional_vars=['input_note_id'], ifdup='error')
flawful.dupkey(df, by_vars=['input_note_id'], desc='df',
               additional_vars=['en1','de1'], ifdup='error')

df['note_id'] = df.input_note_id.map(lambda x: 'FLAWFUL_EX1_' + str(x))
df['de3'] = df.de3.map(flawful.german.show_vowel_length)
df['de_pronun'] = df.de_pronun.map(flawful.german.show_vowel_length)
df['de_conf'] = df.de_conf.map(preprocess_de_conf)
df['de1_sortable'] = df.de1.map(flawful.german.make_sortable_str)
res_mc = df.dech.apply(flawful.init_chapter,
                       str_to_chapter=str_to_chapter)
df['chapter'] = [ x['chapter'] for x in res_mc ]
df['chapter_tags'] = [ x['tags'] for x in res_mc ]

# This function does the most work.
# Field information is passed in three or four equal-sized lists (`fields`,
#  `names`, `seps`, `assign_character`), but this may change in the future.
FieldInfo = namedtuple('FieldInfo', ['sep','assign_chapter'])
field_info = {}
field_info['de1'] = FieldInfo(sep=',', assign_chapter=True)
field_info['at1'] = FieldInfo(sep=',', assign_chapter=True)
field_info['sd1'] = FieldInfo(sep=',', assign_chapter=True)
field_info['de3'] = FieldInfo(sep=';', assign_chapter=True)
field_info['dib_sentences'] = FieldInfo(sep=';', assign_chapter=True)
field_info['de_xref'] = FieldInfo(sep=';', assign_chapter=True)
field_info['de_xref_ignore_ch'] = FieldInfo(sep=';', assign_chapter=False)
names = [ x for x in field_info ]
assign_chapters = [ val.assign_chapter for _, val in field_info.items() ]

res_de = [
     flawful.tag_audio_and_markup(audio_dicts=aud_dicts, wordlists=de_dicts,
         str_to_wordlist_key=make_wordlist_key_notes,
         str_to_audio_key=make_audio_key_notes,
         select_keys_no_audio=filter_text_not_audio_pre,
         htag_prefix='DE', chapter=row[0],
         tokens=[ row[1+idx].split(field_info[val].sep)
                  for idx, val in enumerate(names)],
         names=names, assign_chapter=assign_chapters,
         )
     for row in df[['chapter'] + names].values
         ]
# Put each element in `res_de` in own data frame column.
df['de_audio'] = [x.audio_output for x in res_de]
df['de_no_audio'] = np.where((df['de_audio'] == ''), 'no audio', '')
df['chapter'] = [x.chapter for x in res_de]
df['tags'] = [x.tags for x in res_de]
df['tags'] = df['chapter_tags'] + ' ' + df['tags']
for k in de_dicts.keys():
    df[f'In{k}'] = [x.in_wordlists[k] for x in res_de]
for k, val in field_info.items():
    df[k + '_color'] = [val.sep.join(x.markup_output[k]) for x in res_de]
df['de_sentences'] = [flawful.list_of_lists_to_str(x.sent_lists['LC'])
                 for x in res_de]
# done with `res_de`

# We use this as the primary answer for the flashcard, although we could also
# have simply left de1_color, at1_color, and sd1_color in separate
# fields on the back side of the flashcard.
df['de1_at1_sd1_color'] = np.select(
   [ (df.at1 != '') & (df.at1 == df.sd1),
     (df.at1 != '') & (df.sd1 != ''),
     (df.at1 != ''),
     (df.sd1 != '')],
   [ df.de1_color + '; A/CH: ' + df.at1_color,
     df.de1_color + '; A: ' + df.at1_color + '; CH: ' + df.sd1_color,
     df.de1_color + '; A: ' + df.at1_color,
     df.de1_color + '; CH: ' + df.sd1_color], default=df.de1_color)

df['de_notes_list'] = df.de_notes.map(lambda x: x.split(';'))
df['de1_list'] = df.de1.map(lambda x: x.split(','))
df['de2_list'] = [ tokenize_de2(de1=row[0], de2=row[1], part_of_speech=row[2])
                   for row in df[['de1','de2','part_of_speech']].values]
df['de2_problems'] = [
         flawful.german.check_de2_problems(de1=row[0], de2=row[1],
                                           part_of_speech=row[2])
         for row in df[['de1','de2','part_of_speech']].values
                 ]

# Make a string with the number of words in the answer, e.g.
# '1/2 + A:1' means there are two answers in the de1 field (one is required
# and the other optional), and one answer in the at1 field.
df['de_target_number'] = [
          flawful.german.make_target_prompt(de1=row[0], sep=',', flags=FLAGS,
                                            at1=row[1], sd1=row[2])
          for row in df[['de1','at1','sd1']].values
          ]
df['has_german_audio'] = df['de_audio'] != ''

#------------------------------------------------------------------------------
# The fields created above are sufficient, but we would like to go a step
# further and put the prompts and answers in HTML format.
#------------------------------------------------------------------------------
df['n_de1'] = df.de1.map(flawful.count_tokens)
df['n_de3'] = df.de3.map(flawful.count_tokens)
df['n_de3_prompt'] = df.de3_prompt.map(flawful.count_tokens)
df['n_match'] = np.where( df.n_de3 == df.n_de3_prompt, 'Y', 'N')
df['de1_prompt'] = (df.en1 + ' (' + df.part_of_speech + ') '
                           + df.de_target_number)
df['de1_prompt'] += np.where(df.de1_hint != '', ' [' + df.de1_hint + ']', '')
make_rv = [
    flawful.make_prompt_and_answer_table(
            prompts=[r[0],''], answers=[r[1],r[2]],
            expr_prompts=r[3].split(';'),
            expr_answers=r[4].split(';'),
            drop_empty_rows=True)
    for r in df[['de1_prompt','de1_at1_sd1_color','de2','de3_prompt',
                 'de3_color']].values
          ]
df['de_table_prompt'] = [ x['prompt'] for x in make_rv ]
df['de_table_answer'] = [ x['answer'] for x in make_rv ]
df['de3_omitted'] = [ x['exprs_omitted'] for x in make_rv ]

# Make table where German words are in the first column
make_rev_rv1 = [
    flawful.make_prompt_and_answer_table(
        prompts=[r[1],''], answers=[r[0],''],
        expr_prompts=r[4].split(';'),
        expr_answers=r[3].split(';'),
        drop_empty_rows=True)
   for r in df[['de1_prompt','de1_at1_sd1_color','de2','de3_prompt',
                'de3_color']].values
          ]
df['de_rev_table_prompt'] = [ x['prompt'] for x in make_rev_rv1 ]
df['de3_rev_omitted'] = [ x['exprs_omitted'] for x in make_rev_rv1 ]

# The 'answer' side for the above. Only difference is `de2` is put
# in the first column of the second row.
make_rev_rv2 = [
    flawful.make_prompt_and_answer_table(
        prompts=[r[1],r[2]], answers=[r[0],''],
        expr_prompts=r[4].split(';'),
        expr_answers=r[3].split(';'),
        drop_empty_rows=True)
   for r in df[['de1_prompt','de1_at1_sd1_color','de2','de3_prompt',
                'de3_color']].values
          ]
df['de_rev_table'] = [ x['answer'] for x in make_rev_rv2 ]

# Suppose we decide ahead of time that for each note we only want to study one
# of the cards, then we can make a tag indicating which side we want to keep.
# We can then easily suspend the other card in Anki. We could extend this by
# also having a tag 'StudyBoth' or by having a column in the input txt file
# that can override this rule, etc...
df['tags'] = df.tags + np.where((df.chapter < 5) | (df.n_de1 > 1),
                                ' StudyEN', ' StudyDE')

#print(flawful.twowaytbl(df, 'n_de3','n_de3_prompt'))

#------------------------------------------------------------------------------
# Optional code to make additional output file
#------------------------------------------------------------------------------
for de3, de3_prompt in df[['de3','de3_prompt']].values:
    flawful.add.check_flag_usage(de3.split(';'), de3_prompt.split(';'),
                     flags=DE3_FLAGS_TO_CHECK)

if DE_ADDITIONAL_INPUT_FILE is not None:
    de_override_df = pd.read_csv(os.path.join(INPUT_DIR,
                                              DE_ADDITIONAL_INPUT_FILE),
                                 sep='\t', skiprows=(0), na_filter=False,
                                 dtype={'id': str}, quoting=csv.QUOTE_NONE)
    de_override_df['id'] = 'AD_' + de_override_df.id
    de_override_df['pronun'] = de_override_df.pronun.map(
            flawful.german.show_vowel_length)
    de_override_df['en_answer_list'] = de_override_df.en_answer.map(
            lambda x: x.split(';'))
    de_override_df['de_answer_list'] = de_override_df.de_answer.map(
            lambda x: x.split(';'))
    if DE_ADD_INPUT_MAPPER is not None:
        de_override_df.rename(columns=DE_ADD_INPUT_MAPPER, inplace=True)
else:
    de_override_df = None

if ADD_INPUT_MAPPER is not None:
    df_mod = df.rename(columns=ADD_INPUT_MAPPER)
else:
    df_mod = df

add_df = flawful.add.create_tl_additional_output(df_mod,
                 aud_dicts=aud_dicts, wordlists=de_dicts,
                 str_to_wordlist_key=make_wordlist_key_notes,
                 str_to_audio_key=make_audio_key_notes,
                 select_keys_no_audio=filter_text_not_audio_pre,
                 tl_override_df=de_override_df,
                 str_to_chapter=str_to_chapter,
                 nl_abbr='E', tl_abbr='D', htag_prefix='DE',
                 braces_html_class=BRACES_HTML_CLASS,
                 flags=FLAGS)

if DE_ADD_OUTPUT_MAPPER is not None:
    add_df.rename(columns=DE_ADD_OUTPUT_MAPPER, inplace=True)

add_outfile = os.path.join(OUTPUT_DIR, DE_ADD_OUTPUT_FILE_PREFIX)
add_df.to_csv(f'{add_outfile}.txt', sep='\t', quoting=csv.QUOTE_NONE,
              index=False, header=False)
add_df[0:0].to_csv(f'{add_outfile}_fields.txt',
                   sep='\t', quoting=csv.QUOTE_NONE, index=False)

#------------------------------------------------------------------------------
# Print words in various external lists that were not in the input notes
#------------------------------------------------------------------------------
de_dicts.print_unused_words(os.path.join(OUTPUT_DIR,
                                       'wordlists_headwords_not_in_notes.txt'),
                            WORDLISTS_TO_COMPARE, de_okay_dicts)

# Copy audio files to production directory and/or list unused files.
if COPY_AUDIO:
    audio_file_dict.copy_used_files(AUDIO_OUTPUT_DIR)
if PRINT_UNUSED_AUDIO:
    audio_file_dict.print_unused_audio()

# Never sort when exporting all records, because the typical use case in
# that scenario is to create some new column that is then copied to the
# source spreadsheet. Otherwise, it's probably sensible to sort by chapter.
if not EXPORT_ALL_RECORDS:
    df = df.sort_values(['chapter','de1_sortable','en1','note_id'])

#dfout = df[(df.chapter <= MAX_CHAPTER)]
dfout = select_output_rows(df, MAX_CHAPTER, ONE_CHAPTER, EXPORT_ALL_RECORDS)

make_tables_and_listings(dfout, WORDLISTS_TO_COMPARE,
                         PRINT_NOTES_WITHOUT_AUDIO, ONE_CHAPTER)

write_de2_problems(dfout, os.path.join(OUTPUT_DIR, 'de2_problems.txt'))

if WRITE_WORDS_WITHOUT_AUDIO:
    flawful.german.write_keys_no_audio(
        os.path.join(OUTPUT_DIR, 'words_no_audio.txt'),
        aud_dicts['keys_no_audio'], filter_text_not_audio_post)

de_dicts.compare(dict_ids=WORDLISTS_TO_COMPARE)

#------------------------------------------------------------------------------
# Select fields for output and output
#------------------------------------------------------------------------------
dfout = select_output_columns(dfout)
dfout.to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}.txt'), sep='\t',
             index=False, header=False, quoting=csv.QUOTE_NONE)
# make an empty dataset and just write the column names
dfout[0:0].to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}_fields.txt'),
                  sep='\t', index=False, quoting=csv.QUOTE_NONE)

#------------------------------------------------------------------------------
# Alternate output that adds metadata to the file header
#------------------------------------------------------------------------------
# If using this method, the field names used in Anki must match the ones used
# in the Python program (since they are being put in the file header).
# Then, when selecting 'File' > 'Import' in Anki, the defaults in the import
# dialog will be set as specified in the header.
#
# Users could also write a simple add-on to add a menu item that imports the
# file automatically, without any selection, for example, by combining code
# at the first two links below. For details, see:
#
#   https://addon-docs.ankiweb.net/a-basic-addon.html
#   https://addon-docs.ankiweb.net/the-anki-module.html
#   https://docs.ankiweb.net/importing/text-files.html
#
# The last link has most of the values for the metadata, but the '#if matches'
# tag can (at least in v24.06.03) take values {'update current',
# 'keep current', 'keep both'} (without the single quotes). No official
# support is available from Anki for add-on writing (or from authors of this
# package of course). See https://addon-docs.ankiweb.net/support.html for
# support suggestions. Test any add-ons carefully on any Anki version they
# will be run on.
#------------------------------------------------------------------------------
#dfout = select_output_columns(dfout)
#tags_col = dfout.columns.get_loc('tags')
#tab_separated_column_names = '\t'.join(dfout.columns)  # pylint: disable=invalid-name
#metadata = ['#separator:Tab',
#            '#deck:Flawful_Example_1',
#            '#notetype:Flawful_Example_1',
#            '#html:true',
#           f'#tags column:{tags_col + 1}',
#            '#if matches:update current',
#           f'#columns:{tab_separated_column_names}',
#           ]
#metadata_df = pd.DataFrame(metadata)
## This data frame only has one column, so we just pick a delimiter besides tab
## that is not in the data frame anywhere.
#metadata_df.to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}.txt'),
#              sep='@', index=False, header=False, quoting=csv.QUOTE_NONE)
#dfout.to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}.txt'), sep='\t',
#             index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
#dfout[0:0].to_csv(os.path.join(OUTPUT_DIR,
#                               f'{OUTPUT_FILE_PREFIX}_fields.txt'),
#                  sep='\t', index=False, quoting=csv.QUOTE_NONE)
