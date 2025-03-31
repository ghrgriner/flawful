#    Functions to create additional cards with German prompts for study.
#    Copyright (C) 2025 Ray Griner (rgriner_fwd@outlook.com)

""" Functions to create additional cards with German prompts for study.

They are additional in the sense that some cards with German prompts are
already created by reversing cards with English prompts for notes in the
primary input file. The additional cards are from one of two sources:
(1) flagged words from `de1` in the primary input file (described below),
and (2) from notes entered in a second input text file.

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
raw data for these notes in the secondary input file.
"""

#------------------------------------------------------------------------------
# File:   de_additional.py
# Date:   2025-03-16
# Author: Ray Griner
#------------------------------------------------------------------------------

import csv
import pandas as pd
import numpy as np
import flawful

#---------------------------------------------
# Functions
#---------------------------------------------
def check_flag_usage(col1, col2, flags):
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
        List of tokens from the first column of the table.
    col2 : str
        List of tokens from the second column of the table.
    flags : str
        Flags. Each character in the string is a unique flag.

    Returns
    -------
    Boolean indicator whether a warning was generated.
    """
    col1_list = col1
    col2_list = col2
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
                   flags, en1, part_of_speech, en1_hint, de1_hint,
                   de2, de1_list, de2_list, de_notes, de_pronun):
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
    flag_set = set(flags)
    for idx, val in enumerate(de1_list):
        val = val.strip()
        headword = str_to_wordlist_key(val)
        if not flag_set.isdisjoint(val):
            if headword in exclude_headwords:
                continue

            # get the token with the same index from `de2`
            try:
                de2 = de2_list[idx]
            except IndexError:
                de2 = ''

            # Make dictionary containings definitions. Key is index, starting
            # at 1 instead of 0.
            def_dict = to_def_dict(de_notes, sep=';')
            definition = ''
            def_type = ''
            if idx + 1 in def_dict:
                if def_dict[idx+1][0]:
                    def_type = f'[{de1_hint}]'
                    # def_dict value = (M, 'some text' | '')
                    definition = (de1_list[def_dict[idx+1][0]-1] + ' '
                                  + def_dict[idx+1][1])
                else:
                    # def_dict value = (None, 'some text')
                    def_type = f'[{en1_hint}|{de1_hint}]'
                    definition = def_dict[idx+1][1]
            else:
                # Not in dictionary. The primary answer will be `en1`, but
                # this will also be on the back of the card, so we don't
                # put it in `definition`.
                def_type = f'[{en1_hint}]'
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

            dict_val = {'merge_id': 'HW_' + headword,
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

def process_de_override_df(df, aud_dicts, wordlists, str_to_chapter,
                 str_to_wordlist_key, str_to_audio_key,
                 braces_html_class, en_hint, de_hint, htag_prefix,
                 select_keys_no_audio):
    """Process `de_override_pf` passed to `create_de_additional_output`.

    Parameters
    ----------
    The `df` parameter to this function is the `de_override_df` parameter
    to `create_de_additional_output`. All parameters are passed through
    from `create_de_additional_output`. See that function for details.

    Returns
    -------
    A data frame (pd.DataFrame) with the same number of observations as the
    input `df`. The columns are:
       'id','merge_id','de_table_answer','de_table_prompt','pronun','notes',
       'audio','chapter','Tags'.
    """

    # TODO: for now, match the fact that we originally built `de3_prompts`
    # by '; '.join(...), in other words, to replicate we need to add a space
    # after the first value in the list.
    def add_space(list_):
        return [ (' ' if idx > 0 else '') + val
                 for idx, val in enumerate(list_) ]

    if ('de3_prompts_list' not in df and 'de3_list' not in df):
        df['de3p'] = flawful.columns_with_prefix_to_list(df, 'de3p_')
        df['de3d'] = flawful.columns_with_prefix_to_list(df, 'de3d_')
        df['de3e'] = flawful.columns_with_prefix_to_list(df, 'de3e_')
        ret_val = [ flawful.combine_answer_lists(prompts=de3p, answers_1=de3d,
                    answers_2=de3e, answer1_hint=de_hint,
                    answer2_hint=en_hint)
                for (de3p, de3d, de3e) in df[['de3p','de3d','de3e']].values
                  ]
        #df['de3_prompts_list'] = [ x['prompts'] for x in ret_val ]
        #df['de3_list'] = [ x['answers'] for x in ret_val ]
        df['de3_prompts_list'] = [ add_space(x['prompts']) for x in ret_val ]
        df['de3_list'] = [ add_space(x['answers']) for x in ret_val ]
    elif ('de3_prompts_list' in df or 'de3_list' in df):
        raise ValueError('Both or none of `de3_prompts_list` and `de3_list`'
                         'columns should be present in `df`.')

    def braces_to_class_list(x):
        return [ flawful.braces_to_class(val, html_class=braces_html_class)
                 for val in x ]
    if braces_html_class is not None:
        df['de3_prompts_list'] = df.de3_prompts_list.map(braces_to_class_list)
        df['de3_list'] = df.de3_list.map(braces_to_class_list)

    ret_mtp = [
         flawful.make_hint_target_and_answer(
                     answer1=de_answer, answer2=en_answer,
                     answer1_hint=de_hint,  answer2_hint=en_hint,  sep=';')
         for (de_answer, en_answer) in df[['de_answer','en_answer']].values
              ]
    df['target'] = [ x['hint'] + ': ' + x['target'] for x in ret_mtp ]
    df['answer'] = [ x['answer'] for x in ret_mtp ]
    df['de_for_headword'] = np.where(df.de_xref != '', df.de_xref, df.de1)
    df['merge_id'] = 'HW_' + df.de_for_headword.map(str_to_wordlist_key)
    res_mc = df.chaplist.apply(flawful.init_chapter,
                               str_to_chapter=str_to_chapter)
    df['min_chap'] = [ x['chapter'] for x in res_mc ]
    df['chap_tags'] = [ x['tags'] for x in res_mc ]

    res_de1 = [
          flawful.tag_audio_and_markup(audio_dicts=aud_dicts,
                 wordlists=wordlists,
                 str_to_wordlist_key=str_to_wordlist_key,
                 str_to_audio_key=str_to_audio_key,
                 select_keys_no_audio=select_keys_no_audio,
                 htag_prefix=htag_prefix,
                 chapter=row[1],
                 tokens=[[row[0]]],
                 names=['de1'],
                 assign_chapter=[True])
          for row in df[['de1','min_chap']].values
              ]
    df['de1_color'] = [''.join(x.markup_output['de1']) for x in res_de1]
    df['audio'] = [x.audio_output for x in res_de1]
    df['chapter'] = [x.chapter for x in res_de1]
    df['Tags'] = [x.tags for x in res_de1]
    df['Tags'] = df['chap_tags'] + ' ' + df['Tags']
    df['prompt'] = (df.de1_color + ' (' + df.part_of_speech + ') '
                       + df.target)
    df['prompt'] = np.where(df.de1_hint == '', df.prompt,
                            df.prompt + ' [' + df.de1_hint + ']')

    make_rv = [
        flawful.make_prompt_and_answer_table(
            prompts=[r[0],''], answers=[r[1],''],
            expr_prompts=r[2], expr_answers=r[3],
            drop_empty_rows=True)
       for r in df[['prompt','answer','de3_prompts_list', 'de3_list']].values
              ]
    df['de_table_prompt'] = [ x['prompt'] for x in make_rv ]

    # rerun to make answer, only difference is `prompts` parameter
    make_rv = [
        flawful.make_prompt_and_answer_table(
            prompts=[r[0],r[4]], answers=[r[1],''],
            expr_prompts=r[2], expr_answers=r[3],
            drop_empty_rows=True)
       for r in df[['prompt','answer','de3_prompts_list',
                    'de3_list','de2']].values
              ]
    df['de_table_answer'] = [ x['answer'] for x in make_rv ]

    df = df[['id','merge_id','de_table_answer', 'de_table_prompt','pronun',
       'notes','audio','chapter','Tags']]

    return df

def create_de_additional_output(df, outfile, aud_dicts, wordlists,
                 str_to_wordlist_key,
                 str_to_audio_key,
                 select_keys_no_audio,
                 flags,
                 sep = ',',
                 de_override_df = None,
                 str_to_chapter = None,
                 braces_html_class = None,
                 en_hint = 'E',
                 de_hint = 'D',
                 htag_prefix = 'DE',
                 output_mapper = None,
                              ):
    """Create output file for `DE additional` notes.

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
        - de3_list : Expressions
        - de3_prompts_list : Prompts for expressions
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
    de_override_df : pd.DataFrame, optional
        Additional input data frame for generating notes / cards where the
        German word is on the front. This will override any cards with the
        same headword generated from `df`. This is done because the cards
        generated from `df` have some limitations. The front side of the
        card is basically just what is extracted from `de1`, and there was
        no easy way to add hints or a table of other prompts for
        expressions containing the headword.

        The required columns (in any order) are:
        - id : A unique number or string for each input row
        - de1 : Similar meaning as main file, except this should be a
                single word or phrase and not a token-delimited list
        - de_xref : If populated, will generate the headword for merging to
                the flagged records. Otherwise, `de1` will be used.
        - de2 : Similar meaning as on main file
        - de_answer : The meaning(s) of `de1` in German, if available. This
                can be a semi colon delimited list. The number of items in
                the list is only relevant when creating the prompt, so the
                user knows how many meanings they are expected to produce.
        - en_answer : The meaning(s) of `de1` in English. At least one of
                `de_answer` or `en_answer` must be populated.
        - notes : Same meaning as `de_notes` in the main file, except
                unlike the main file, this field will never be parsed for
                synonyms.
        - pronun : Same meaning as `de_pronun` in the main file
        - de1_hint : Same meaning as in the main file.
        - part_of_speech : Same meaning as in the main file.
        - chaplist : Same meaning as in the main file.

        In addition, fields can exist that are analagous to `de3` and
        `de3_prompts` in the main file. These should either be of the form:
           - de3_list : Same meaning as `de3` in the main file, except
             is a list of the tokens instead of a string.
           - de3_prompts_list : Same meaning as `de3_prompts` in the main
             file, except is a list of the tokens instead of a string.
        Alternatively, users can use fields that will be used to generate
        `de3_list` and `de3_prompts_list`. In this case, the input fields
        should each contain a single token.
           - de3p_N : The Nth token for `de3_prompts_list`
           - de3d_N : The Nth token for `de3_list` (in German)
           - de3e_N : The Nth token for `de3_list` (in English)
    str_to_chapter : Callable[[str], int], optional
        Function to convert strings in `de_override_df.chaplist` to integer
        representing the minimum chapter. Passed to `init_chapter()`. See
        that function for details. Must be populated if `de_override_df` is
        not None.
    braces_html_class : str, optional
        If populated, when processing `de_override_df` (if applicable),
        '{text}' in the `de3_list` or `de3_prompts_list` tokens is
        converted to: '<div class={braces_html_class}>text</div>'.
    output_mapper : optional
        If not `None`, `pd.DataFrame.rename` will be called on the dataset
        that makes the additional output file just before the output is
        written, and this will be passed as the mapper. This is because the
        default names generated have `en` prefixes (meaning English) and
        `de` prefixes (meaning German). Users can then store words in other
        languages in these fields and name them something appropriate.
    en_hint : str, optional (default='E')
        The additional file could have answers in `de_answer`, `en_answer`
        or both. The card types derived from flagged words in `de1` could
        have an answer also from `de1`, `en1`, or `de_notes`. This hint is
        added to the prompt created on the front of the card indicating
        from which field the answer was taken.
    de_hint : str, optional (default='D')
        See `en_hint` above.
    htag_prefix : str, optional (default='DE')
        Eventually passed to `flag_audio_and_markup`.

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

    for  (en1,    part_of_speech,   de2,   de_notes,   de_pronun,
          de1_list,    de2_list) in df[
        ['en1', 'part_of_speech', 'de2', 'de_notes', 'de_pronun',
         'de1_list', 'de2_list']].values:
        make_new_cards(exclude_headwords=de1_not_flagged_set,
                       de1_flagged_dict_=de1_flagged_dict,
                       str_to_wordlist_key=str_to_wordlist_key,
                       flags=flags,
                       en1_hint=en_hint, de1_hint=de_hint,
                       en1=en1, part_of_speech=part_of_speech,
                       de2=de2, de1_list=de1_list,
                       de2_list=de2_list, de_notes=de_notes,
                       de_pronun=de_pronun)
    de1_df = pd.DataFrame.from_dict(de1_flagged_dict, orient='index')

    res_de1 = [
          flawful.tag_audio_and_markup(audio_dicts=aud_dicts,
                 wordlists=wordlists,
                 str_to_wordlist_key=str_to_wordlist_key,
                 str_to_audio_key=str_to_audio_key,
                 select_keys_no_audio=select_keys_no_audio,
                 htag_prefix=htag_prefix,
                 chapter=999,
                 tokens=[[row[0]]],
                 names=['de1'],
                 assign_chapter=[True])
          for row in de1_df[['de1']].values
              ]
    de1_df['de_audio'] = [x.audio_output for x in res_de1]
    de1_df['de1_color'] = [''.join(x.markup_output['de1']) for x in res_de1]
    de1_df['chapter'] = [x.chapter for x in res_de1]
    de1_df['Tags'] = [x.tags for x in res_de1]
    vars_in_output = ['note_id', 'en1', 'part_of_speech', 'de_defs',
                     'def_type', 'de1', 'de2', 'de_pronun', 'de_audio',
                     'de1_color', 'chapter']

    if de_override_df is not None:
        df2 = process_de_override_df(df=de_override_df,
                     str_to_chapter=str_to_chapter,
                     aud_dicts=aud_dicts, wordlists=wordlists,
                     str_to_wordlist_key=str_to_wordlist_key,
                     str_to_audio_key=str_to_audio_key,
                     braces_html_class=braces_html_class,
                     select_keys_no_audio=select_keys_no_audio,
                     en_hint=en_hint, de_hint=de_hint,
                     htag_prefix=htag_prefix,
                     )
        df2 = df2.rename({'audio': 'o_audio', 'chapter': 'o_chapter',
                    'Tags': 'o_Tags'}, axis='columns')

        de1_df = de1_df.merge(df2[['id','merge_id','de_table_answer',
           'de_table_prompt','pronun','notes','o_audio','o_chapter','o_Tags']],
            how='outer', on='merge_id', indicator='merge_ind')

        in_df2 = de1_df.merge_ind != 'left_only'
        de1_df['has_table'] = np.where(in_df2, 'has_table', '')
        de1_df['no_table']  = np.where(in_df2, '', 'Y')
        de1_df['de_audio'] = np.where(in_df2, de1_df.o_audio,  de1_df.de_audio)
        de1_df['chapter']  = np.where(in_df2, de1_df.o_chapter, de1_df.chapter)
        de1_df['Tags']     = np.where(in_df2, de1_df.o_Tags,   de1_df.Tags)
        de1_df['note_id']  = np.where(in_df2,
                                      'AD_' + de1_df.id, de1_df.merge_id)
        vars_in_output.extend(['de_table_answer','de_table_prompt','has_table',
                               'no_table','pronun','notes'])

    #de1_df['dummy'] = True
    #print(flawful.twowaytbl(de1_df, 'chapter', 'dummy', cumulative=True))
    de1_df = de1_df[vars_in_output + ['Tags']]
    if output_mapper is not None:
        de1_df.rename(columns=output_mapper, inplace=True)

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
    #           f'#deck:DE Additional',
    #            "#notetype:DE Additional",
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
