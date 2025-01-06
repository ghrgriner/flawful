#    Example 1 (German) language-learning flashcards
#    Copyright (C) 2024 Ray Griner (rgriner_fwd@outlook.com)
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

import pandas as pd
import re
import numpy as np
import flawful
import flawful.german
import os

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
MAX_CHAPTER = 20

# Regular expression for strings where we don't want to remove the initial
# definite article(s) in the token when making the lookup key / headword
KEEP_INITIAL_DER_DAS_DIE_PATTERN = re.compile(
        '^die Niederlande|^der Esel sagt ihm')

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def select_output_columns(df_):
    """Select output columns from the data frame.

    It's sometimes useful to pass in the `EXPORT_ALL_RECORDS` variable
    as a parameter, so we have made this a function instead of a line
    of code.
    """
    return df_[['note_id','en1','part_of_speech','de_target_number',
           'de1','de1_at1_sd1_color','de2','de3','de3_color','de_xref_color',
           'de_en_add','de_conf','de_pronun','dib_sentences',
           'at1','sd1','de_audio','de_no_audio',
           'chapter','de_sentences','tags']]

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
    if ret_val.strip() == 'sich' and x.strip() != 'sich': ret_val = ''
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
            if (not line) or ignore_line_pattern.match(line): continue
            dictkey=make_headword_reflist(line)
            if dictkey in ret_dict:
                # can be changed to just print a message if needed
                raise ValueError(f'Duplicate key in reflist LA: {dictkey}')
            else:
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
        comment='#', sep='\t', na_filter=False, quoting=3,
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
        comment='#', sep='\t', na_filter=False, quoting=3)

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
                         na_filter=False, quoting=3, names=['Word'])
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
    elif x.strip().startswith('F'):
        return 10 + int(x.strip()[1:2])
    else:
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
            comment='#', sep='\t', skiprows=(0), na_filter=False, quoting=3)
    flawful.dupkey(df_, ['Word'], desc='known_no_audio.txt')
    #df['Reason'] = 'not found'
    df_.set_index('Word', inplace=True)
    return df_['Reason'].to_dict()

def make_tables_and_listings(df: pd.DataFrame, wordlist_id,
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
    dfprint = df[['note_id','en1','de1']]
    if len(dfprint[(df.de1=='')])>0:
        print('\nWARNING: Records with de1 missing:')
        print(dfprint[(df.de1=='')])

    #--------------------------------------------------------------------------
    # Cross tabs of Chapter by Audio file status, etc...
    #--------------------------------------------------------------------------
    flawful.twowaytbl(df, title='\nNotes with German audio by chapter:',
              row='chapter', col='has_german_audio')
    if wordlist_id and wordlist_id != 'All':
        flawful.twowaytbl(df, title=('\nNotes by German chapter and status in '
                  f'{wordlist_id} word list:'),
                  row='chapter', col=f'In{wordlist_id}', cumulative=True)

    #--------------------------------------------------------------------------
    # Print records from each unit with no German audio for any field.
    #--------------------------------------------------------------------------
    if print_notes_without_audio:
        if not one_chapter:
            dfprob = df[ (df.de_audio == '') ]
            dfprob = dfprob[['note_id','en1','de1','de3']]
            num_miss = len(dfprob.index)
            print(f'\nMissing German audio ({num_miss} records):')
        else:
            dfprob = df[ (df.chapter == one_chapter)
                            & (df.de_audio == '') ]
            dfprob = dfprob[['note_id','en1','de1','de3']]
            num_miss = len(dfprob.index)
            print(f'\nMissing German audio ({num_miss} records):')
            if num_miss>0: print(dfprob)

    # other examples of descriptive tables
    #df['FirstLB'] = (df['InLB'] & ~df['InLA'])
    #print(df.pivot_table(index=['FirstLB','InLB','InLA'], values='de1',
    #      aggfunc=len, margins=True))

def write_de2_problems(df, outfile):
    """Example function that writes records flagged with problems in `de2`.
    """
    dfout = df[['chapter','de1','de2','de2_problems']]
    dfout = dfout[~(dfout.de2_problems == '')]
    dfout.to_csv(outfile, sep='\t', index=False, quoting=3)
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
                 skiprows=(0), na_filter=False, quoting=3)

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
res_de = [
     flawful.tag_audio_and_markup(audio_dicts=aud_dicts, wordlists=de_dicts,
         str_to_wordlist_key=make_wordlist_key_notes,
         str_to_audio_key=make_audio_key_notes,
         select_keys_no_audio=filter_text_not_audio_pre,
         htag_prefix='DE',
         chapter=row[0],
         fields=[row[1],row[2],row[3],row[4],row[5],         row[6]],
         names= ['de1', 'at1', 'sd1', 'de3','dib_sentences','de_xref'],
         seps=  [','  , ','  , ','  , ';'  ,';'            ,';'      ],
         assign_chapter=[True, True, True, True, True, False],
         )
     for row in df[['chapter','de1','at1','sd1','de3','dib_sentences',
                    'de_xref']].values
         ]
# Put each element in `res_de` in own data frame column.
df['de_audio'] = [x.audio_output for x in res_de]
df['de_no_audio'] = np.where((df['de_audio'] == ''), 'no audio', '')
df['chapter'] = [x.chapter for x in res_de]
df['tags'] = [x.tags for x in res_de]
df['tags'] = df['chapter_tags'] + ' ' + df['tags']
for k in de_dicts.keys():
    df[f'In{k}'] = [x.in_wordlists[k] for x in res_de]
df['de1_color'] = [x.markup_output['de1'] for x in res_de]
df['de3_color'] = [x.markup_output['de3'] for x in res_de]
df['at1_color'] = [x.markup_output['at1'] for x in res_de]
df['sd1_color'] = [x.markup_output['sd1'] for x in res_de]
df['de_xref_color'] = [x.markup_output['de_xref'] for x in res_de]
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

df['de2_problems'] = [
         flawful.german.check_de2_problems(de1=row[0], de2=row[1],
                                           part_of_speech=row[2])
         for row in df[['de1','de2','part_of_speech']].values
                 ]

# Make a string with the number of words in the answer, e.g.
# '1/2 + A:1' means there are two answers in the de1 field (one is required
# and the other optional), and one answer in the at1 field.
df['de_target_number'] = [
          flawful.german.make_target_prompt(de1=row[0], sep=',',
                                            at1=row[1], sd1=row[2])
          for row in df[['de1','at1','sd1']].values
          ]
df['has_german_audio'] = df['de_audio'] != ''

#------------------------------------------------------------------------------
# Print words in various external lists that were not in the input notes
#------------------------------------------------------------------------------
de_dicts.print_unused_words(os.path.join(OUTPUT_DIR,
                                       'wordlists_headwords_not_in_notes.txt'),
                            WORDLISTS_TO_COMPARE, de_okay_dicts)

# Copy audio files to production directory and/or list unused files.
if COPY_AUDIO: audio_file_dict.copy_used_files(AUDIO_OUTPUT_DIR)
if PRINT_UNUSED_AUDIO: audio_file_dict.print_unused_audio()

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
             index=False, header=False, quoting=3)
# make an empty dataset and just write the column names
dfout[0:0].to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}_fields.txt'),
                  sep='\t', index=False, quoting=3)
