#    Functions to create notes for tokens flagged in `de1` (primary answer).
#    Copyright (C) 2025 Ray Griner (rgriner_fwd@outlook.com)

""" Functions to create notes for tokens flagged in `de1` (primary answer).

The general idea is that we study some cards where English is on the front,
and the target language (German in this case) is on the back. In our notes,
the primary answer is in `de1`, which is the translation(s) of `en1`.
Secondary information like verb forms and noun plurals is in `de2`, and
expressions and their English translation (or some other prompt) are in the
fields `de3` and `de3_prompt`. These last two fields are delimited, where
each token will be converted to a row in a table on the generated card.

The point is that a given English card may have more than one entry in
`de1`. However, we do not think it is particularly useful to make ourselves
remember all possible synonyms for a given English word. Instead, if there
are a lot of synonyms, we mark the less common words with a character that
we call a flag (e.g., '°' or '†'). When a word is flagged, we allow the
card to be considered 'right' even we can't produce the flagged word(s).

We still want to ensure we have receptive knowledge of a flagged word, so
this file has code that makes notes (that will be turned into flashcards)
where the flagged tokens from `de1` are the front side of the card.

The back side of the card is obtained by parsing the field containing notes
to look for tokens that are formatted in a certain way (see `to_def_dict`
for details) or by defaulting to `en1`. This last part is perhaps
unnecessarily complex, and users may want to consider simply keeping the
raw data for these notes in a separate input file and processing that file
similarly to the main file processed in `example1.py`.
"""

#------------------------------------------------------------------------------
# File:   de1_flagged.py
# Date:   2025-03-16
# Author: Ray Griner
#------------------------------------------------------------------------------

import csv
import re
import pandas as pd
import numpy as np
import flawful
import flawful.german

#---------------------------------------------
# Functions
#---------------------------------------------
def check_flag_usage(col1, col2, flags, sep):
    """Print warning messages if flags aren't in both columns or if ~ is used.

    The idea is that since the columns in the table can be reversed, we
    want the flag in both columns. The general expectation is that if a
    column is flagged, the flag should be at the end of the token, but
    this isn't checked.

    The function also prints messages if '~' is used in a flagged cell.
    This is a problem because we may generate cards automatically for the
    flagged rows where the primary question is a cell in the row. In this
    case, there is nothing for the '~' to refer back to.

    Parameters
    ----------
    col1 : str
        First column of the table, rows delimited by `sep`.
    col2 : str
        Second column of the table, rows delimited by `sep`.
    flags : str
        Flags. Each character in the string is a unique flag.
    sep : str
        Delimiter used for `col1` and `col2`.

    Returns
    -------
    Boolean indicator whether a warning was generated.
    """
    col1_list = col1.split(sep)
    col2_list = col2.split(sep)
    found = 0
    if len(col1_list) == len(col2_list):
        for idx, val in enumerate(col1_list):
            for flag in flags:
                if flag in val:
                    if flag not in col2_list[idx]:
                        found += 1
                        raise ValueError(f'{flag} not in col1 and col2: '
                                         + col1 + ' | ' + col2)
                    if '~' in val:
                        raise ValueError('~ in col1: ' + col2)
                    if '~' in col2_list[idx]:
                        raise ValueError('~ in col2: '
                                          + col2_list[idx])
                elif flag in col2_list[idx]:
                    raise ValueError(f'{flag} not in col1 and col2: '
                                     + col1 + ' | ' + col2)

def _add_unflagged_headword_to_set(col, set_, str_to_wordlist_key, flags, sep):
    """Add headwords for all unflagged tokens to the input set.
    """
    col_list = col.split(sep)
    flag_set = set(flags)
    for val in col_list:
        headword = str_to_wordlist_key(val)
        if headword and flag_set.isdisjoint(val):
            set_.add(headword)

def to_def_dict(col, sep):
    """Obtain definitions from `col` and put into a dictionary.

    The `col` field should contain semi-formatted text from which the
    definition can be (eventually) extracted. Tokens are processed if they
    are of the form:
    - 'N: some text'
    - 'N=M'
    - 'N=M some text' (usually some text is in parentheses)
    - 'N≈M' (this and the next bullet is the 'approximately equals symbol')
    - 'N≈M some text' (usually some text is in parentheses)'
    - 'N: some text'

    N and M both refer to the position of words in some other field that
    contains German words using an index that starts at 1 instead of the
    usual Python 0. (In the `make_new_cards` function this 'some other
    field' is `de1`, and for concreteness we will use this name here, but
    note that this function does not use the name of the other field.)

    Returning to the contents of `col`, a token that starts with '3=2'
    means that if we make a card with the third token from `de1` on the
    front side, the primary answer on the back side will be the second
    token from `de1`.

    We do not expect all tokens in `de1` to have an entry in the output.

    Parameters
    ----------
    col : str
        See description above.
    sep : str
        Delimiter for tokens in `col`.

    Returns
    -------
    A dictionary. For each token processed, an item is added to the
    output dictionary with key N and value: (M | None, 'some text' | '').
    """

    def_list = col.split(sep)
    ret_dict = {}
    for val in def_list:
       val = val.strip()
       if (len(val) >= 3 and (val[1] == '=' or val[1] == '≈')
          and val[0].isdigit() and val[2].isdigit()):
          if int(val[0]) in ret_dict:
              raise ValueError(f'{def_list} puts in dict twice')
          if len(val) <= 4:
              ret_dict[int(val[0])] = (int(val[2]), '')
          else:
              ret_dict[int(val[0])] = (int(val[2]), val[4:])
       elif (len(val) >= 3) and val[0].isdigit() and val[1] == ':':
          if int(val[0]) in ret_dict:
              raise ValueError(f'{def_list} puts in dict twice')
          ret_dict[int(val[0])] = (None, val[3:])
    return ret_dict

def make_new_cards(exclude_headwords, de1_flagged_dict_, str_to_wordlist_key,
                   sep, flags, en1, part_of_speech,
                   de1, de2, de_notes, de3, de3_prompt, de_pronun):
    """Add items to dictionary with the fields for the DE1_Flagged cards.

    Returns
    -------
    None. `de1_flagged_dict_` has records added as a side-effect. This
    output dictionary can be converted to a data frame by the calling
    function where each item in the dictionary is a row in the data frame.
    The key of each entry is the headword for the token. The value of the
    entry is another dictionary where the key is the column name and the
    value is the column value.
    """
    transtab = str.maketrans('', '', flags)
    de1_list = de1.split(sep)
    flag_set = set(flags)
    for idx, val in enumerate(de1_list):
        val = val.strip()
        headword = str_to_wordlist_key(val)
        if not flag_set.isdisjoint(val):
            if headword in exclude_headwords:
                continue

            # get the token with the same index from `de2`
            if (('(in)' in de1 and part_of_speech == 'N')
                or part_of_speech == 'V'):
                try:
                    de2 = de2.split(';')[idx]
                except IndexError: de2 = ''
            elif part_of_speech == 'N':
                try:
                    de2 = de2.split(',')[idx]
                except IndexError: de2 = ''
            else:
                de2 = ''

            # Make dictionary containings definitions. Key is index, starting
            # at 1 instead of 0.
            def_dict = to_def_dict(de_notes, sep=';')
            definition = ''
            def_type = ''
            if (idx + 1) in def_dict:
                if def_dict[idx+1][0]:
                    def_type = '[DE]'
                    # def_dict value = (M, 'some text' | '')
                    definition = (de1_list[def_dict[idx+1][0]-1] + ' '
                                  + def_dict[idx+1][1])
                else:
                    # def_dict value = (None, 'some text')
                    def_type = '[EN (or DE)]'
                    definition = def_dict[idx+1][1]
            else:
                # Not in dictionary. The primary answer will be `en1`, but
                # this will also be on the back of the card, so we don't
                # put it in `definition`.
                def_type = '[EN]'
                if f'{idx+1}:' in de_notes or f'{idx+1}=' in de_notes:
                    # might happen when we accidentally used comma instead
                    # of semi-colon to tokenize de3
                    raise ValueError(f'ERROR: {val} incorrect format def |'
                                     f' {idx+1} | {de_notes}')
            if not headword:
                # Empty headwords are permitted for the SPDEFull note type,
                # but not for this note type. If this is raised, change the
                # note so the headword isn't empty.
                raise ValueError('headword empty for val = ' + val)

            dict_val = {'note_id': 'DE1_' + headword,
                        'en1': en1,
                        'part_of_speech': part_of_speech,
                        'de_defs': definition,
                        'def_type': def_type,
                        'de1': val.translate(transtab),
                        'de2': de2,
                        'de_pronun': de_pronun}
            # If headword is duplicated, silently keep the first one entered.
            if headword in de1_flagged_dict_:
                pass
            else:
                de1_flagged_dict_[headword] = dict_val

def create_de1_flagged_output(df, outfile, aud_dicts, wordlists,
                 str_to_wordlist_key,
                 str_to_audio_key,
                 select_keys_no_audio,
                 flags,
                 sep = ',',
                              ):
    """Create output file for `DE1 flagged` notes.

    The `de1` field is parsed and (1) tokens marked with '°' are
    identified.  (2) tokens NOT marked with '°' or '†' are then identified.
    A row in the output will be created for each headword in (1) that is
    not in the set of headwords for (2).

    See module docstring for background.

    A given headword will not be duplicated in the output.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with the following columns:
        - en1 : English word or phrase
        - de1 : Primary German word or phrase(s)
        - de2 : Secondary German word or phrase(s). Semicolon delimited
                when (`de1` contains '(in)' and `part_of_speech == 'N'`)
                or when `part_of_speech == 'V'`.
        - de3 : Expressions
        - de3_prompt : Prompts for expressions
        - de_notes : Notes, including definitions. This is a semi-colon
                formatted string. If a token matches the format in
                `to_note_dict`, it will be processed and definition
                information extracted. Other tokens are ignored.
        - de_pronun : Contains pronunciation information. This is passed
                through to the output file.
    outfile : str
        File name and path for prefix output file. The program will append
        '.txt' to make the name for the file with the notes data and will
        append '_fields.txt' to make the name for the file with the field
        names.
    aud_dicts : Dict
        See `aud_dicts` parameter in flawful.tag_audio_and_markup().
    wordlists : flawful.Wordlist
        See `wordlists` parameter in flawful.tag_audio_and_markup().
    str_to_wordlist_key: function
        See `str_to_wordlist_key` parameter in
        flawful.tag_audio_and_markup()
    str_to_audio_key: function
        See `str_to_audio_key` parameter in flawful.tag_audio_and_markup().
    select_keys_no_audio : function
        See `select_keys_no_audio` parameter in
        flawful.tag_audio_and_markup().
    flags : str
        String containing flags. Each character in the string is considered
        a single flag.
    sep : str
        Delimiter for `de1`.

    Returns
    -------
    None, but `aud_dicts` and `wordlists` updated as side-effects. The
    output file is also written to location `outfile`.
    """

    de1_not_flagged_set = set()
    de1_flagged_dict = {}

    df.de1.map(lambda x: _add_unflagged_headword_to_set(x, de1_not_flagged_set,
                                                        str_to_wordlist_key,
                                                        flags, sep))

    for (en1,    part_of_speech,   de1,   de2,   de_notes,   de3,  de3_prompt,
         de_pronun) in df[
        ['en1', 'part_of_speech', 'de1', 'de2', 'de_notes', 'de3','de3_prompt',
        'de_pronun']].values:
        make_new_cards(exclude_headwords=de1_not_flagged_set,
                       de1_flagged_dict_=de1_flagged_dict,
                       str_to_wordlist_key=str_to_wordlist_key,
                       sep=sep, flags=flags,
                       en1=en1, part_of_speech=part_of_speech,
                       de1=de1, de2=de2, de_notes=de_notes, de3=de3,
                       de3_prompt=de3_prompt, de_pronun=de_pronun)
    de1_df = pd.DataFrame.from_dict(de1_flagged_dict, orient='index')

    res_de1 = [
          flawful.tag_audio_and_markup(audio_dicts=aud_dicts,
                 wordlists=wordlists,
                 str_to_wordlist_key=str_to_wordlist_key,
                 str_to_audio_key=str_to_audio_key,
                 select_keys_no_audio=select_keys_no_audio,
                 htag_prefix='DE',
                 chapter=999,
                 fields=[row[0]],
                 names=['de1'],
                 seps=[';'],
                 assign_chapter=[True])
          for row in de1_df[['de1']].values
              ]
    de1_df['de_audio'] = [x.audio_output for x in res_de1]
    de1_df['de1_color'] = [x.markup_output['de1'] for x in res_de1]
    de1_df['chapter'] = [x.chapter for x in res_de1]
    de1_df['Tags'] = [x.tags for x in res_de1]
    #de1_df['dummy'] = True
    #print(flawful.twowaytbl(de1_df, 'chapter', 'dummy', cumulative=True))
    de1_df = de1_df[['note_id', 'en1', 'part_of_speech', 'de_defs',
                     'def_type', 'de1', 'de2', 'de_pronun', 'de_audio',
                     'de1_color', 'chapter', 'Tags']]

    de1_df.to_csv(f'{outfile}.txt', sep='\t', quoting=csv.QUOTE_NONE,
                  index=False)
    de1_df[0:0].to_csv(f'{outfile}_fields.txt', sep='\t',
                       quoting=csv.QUOTE_NONE, index=False)

    #--------------------------------------------------------------------------
    # Alternate output that adds metadata to the file header
    #--------------------------------------------------------------------------
    # The code that writes to '{outfile}.txt' and f'{outfile}_fields.txt' above
    # can be replaced with the below to add metadata to the file header. See
    # similar section in example1.py for discussion and cautions.
    #
    #tags_col = de1_df.columns.get_loc('Tags')
    #column_str = '\t'.join(de1_df.columns)
    #metadata = ['#separator:Tab',
    #           f'#deck:DE1 Flagged',
    #            "#notetype:DE1 Flagged",
    #            '#html:true',
    #           f'#tags column:{tags_col + 1}',
    #           f"#if matches:update current",
    #           f"#columns:{column_str}",
    #           ]
    #meta_df = pd.DataFrame(metadata)
    #meta_df.to_csv(f'{outfile}.txt', sep='@', index=False, header=False,
    #             quoting=csv.QUOTE_NONE)
    #de1_df.to_csv(f'{outfile}.txt', sep='\t', quoting=csv.QUOTE_NONE,
    #              index=False, mode='a')

