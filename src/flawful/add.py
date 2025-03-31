#    Create additional cards with target language prompts for study.
#    Copyright (C) 2025 Ray Griner (rgriner_fwd@outlook.com)

""" Create additional cards with target language prompts for study.

They are additional in the sense that some cards with target language
prompts are already created by reversing cards with native language prompts
for notes in the primary input file. The additional cards are from one of
two sources: (1) flagged words from `tl1` in the primary input file
(described below), and (2) from notes entered in a second input text file.

The general idea is that we study some cards where the native language is
on the front and the target language is on the back. The primary answer
must be in `tl1`, which is the translation of `nl1`.
Secondary information (which depending on the language might be verb forms
or noun plurals) is in `tl2`, and expressions or other questions the user
wants to answer are in `tl3` and `tl3_prompt`.

The point is that a given native language entry may have more than one word
or phrase associated in `tl1`. However, we do not think it is particularly
useful to make ourselves remember all possible synonyms for a given native
language word. Instead, if there are a lot of synonyms, we mark the less
common words with a character that we call a flag (e.g., '°' or '†'). When
a word is flagged, we allow the card to be considered 'right' even we can't
produce the flagged word(s).

We still want to ensure we have receptive knowledge of a flagged word, so
this file has code that makes notes (that will be turned into flashcards)
where the flagged tokens from `nl1` are the front side of the card.

The back side of the card is obtained by parsing the field containing notes
to look for tokens that are formatted in a certain way (see `to_def_dict`
for details) or by defaulting to `nl1`. This last part is perhaps
unnecessarily complex, and users may want to consider simply keeping the
raw data for these notes in the additional input file.
"""

#------------------------------------------------------------------------------
# File:   add.py
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

def _add_unflagged_headword_to_set(col_list, set_, str_to_wordlist_key, flags):
    """Add headwords for all unflagged tokens to the input set.
    """
    flag_set = set(flags)
    for val in col_list:
        val = val.strip()
        headword = str_to_wordlist_key(val)
        if headword and flag_set.isdisjoint(val):
            set_.add(headword)

def _to_def_dict(def_list):
    """Obtain definitions from `def_list` and put into a dictionary.

    `def_list` should contain tokens from which the definition can be
    (eventually) extracted. Tokens are processed if they are of the form:
    - 'N: some text'
    - 'N=M'
    - 'N=M some text' (usually some text is in parentheses)
    - 'N≈M' (this and the next bullet is the 'approximately equals symbol')
    - 'N≈M some text' (usually some text is in parentheses)'
    - 'N: some text'

    N and M both refer to the position of words in some other field that
    contains target language words using an index that starts at 1 instead of
    the usual Python 0. (In the `make_new_cards` function this 'some other
    field' is `tl1`, and for concreteness we will use this name here, but
    note that this function does not use the name of the other field.)

    Returning to the contents of `col`, a token that starts with '3=2'
    means that if we make a card with the third token from `tl1` on the
    front side, the primary answer on the back side will be the second
    token from `tl1`.

    We do not expect all tokens in `tl1` to have an entry in the output.

    Returns
    -------
    A dictionary. For each token processed, an item is added to the
    output dictionary with key N and value: (M | None, 'some text' | '').
    """

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

def _make_new_cards(exclude_headwords, tl1_flagged_dict_, str_to_wordlist_key,
                   flags, nl1, part_of_speech, nl1_hint, tl1_hint,
                   tl1_list, tl2_list, tl_notes_list, tl_pronun):
    """Add items to dictionary with the fields for the TL Flagged cards.

    Returns
    -------
    None. `tl1_flagged_dict_` has records added as a side-effect. This
    output dictionary can be converted to a data frame by the calling
    function where each item in the dictionary is a row in the data frame.
    The key of each entry is the headword for the token. The value of the
    entry is another dictionary where the key is the column name and the
    value is the column value.
    """
    transtab = str.maketrans('', '', flags)
    flag_set = set(flags)
    for idx, val in enumerate(tl1_list):
        val = val.strip()
        headword = str_to_wordlist_key(val)
        if not flag_set.isdisjoint(val):
            if headword in exclude_headwords:
                continue

            # get the token with the same index from `tl2`
            try:
                tl2 = tl2_list[idx].strip()
            except IndexError:
                tl2 = ''

            # Make dictionary containings definitions. Key is index, starting
            # at 1 instead of 0.
            def_dict = _to_def_dict(tl_notes_list)
            definition = ''
            answer_lang = ''
            if idx + 1 in def_dict:
                if def_dict[idx+1][0]:
                    answer_lang = f'[{tl1_hint}]'
                    # def_dict value = (M, 'some text' | '')
                    definition = (tl1_list[def_dict[idx+1][0]-1].strip() + ' '
                                  + def_dict[idx+1][1])
                else:
                    # def_dict value = (None, 'some text')
                    answer_lang = f'[{nl1_hint}|{tl1_hint}]'
                    definition = def_dict[idx+1][1]
            else:
                # Not in dictionary. The primary answer will be `nl1`, but
                # this will also be on the back of the card, so we don't
                # put it in `definition`.
                answer_lang = f'[{nl1_hint}]'
                tl_notes = ';'.join(tl_notes_list)
                if f'{idx+1}:' in tl_notes or f'{idx+1}=' in tl_notes:
                    # might happen when we accidentally used comma instead
                    # of semi-colon to tokenize tl3
                    raise ValueError(f'ERROR: {val} incorrect format def |'
                                     f' {idx+1} | {tl_notes}')
            if not headword:
                raise ValueError('headword empty for val = ' + val)

            dict_val = {'merge_id': headword,
                        'nl1': nl1,
                        'part_of_speech': part_of_speech,
                        'tl_defs': definition,
                        'answer_lang': answer_lang,
                        'tl1': val.translate(transtab),
                        'tl2': tl2,
                        'tl_pronun': tl_pronun}
            # If headword is duplicated, silently keep the first one entered.
            if headword in tl1_flagged_dict_:
                pass
            else:
                tl1_flagged_dict_[headword] = dict_val

def _process_tl_override_df(df, aud_dicts, wordlists, str_to_chapter,
                 str_to_wordlist_key, str_to_audio_key,
                 braces_html_class, nl_abbr, tl_abbr, htag_prefix,
                 select_keys_no_audio):
    """Process `tl_override_pf` passed to `create_tl_additional_output`.

    Parameters
    ----------
    The `df` parameter to this function is the `tl_override_df` parameter
    to `create_tl_additional_output`. All parameters are passed through
    from `create_tl_additional_output`. See that function for details.

    Returns
    -------
    A data frame (pd.DataFrame) with the same number of observations as the
    input `df`. The columns are:
       'id','merge_id','tl_table_answer','tl_table_prompt','pronun','notes',
       'audio','chapter','Tags'.
    """

    if ('tl3_prompts_list' not in df and 'tl3_list' not in df):
        df['tl3p'] = flawful.columns_with_prefix_to_list(df, 'tl3p_')
        df['tl3t'] = flawful.columns_with_prefix_to_list(df, 'tl3t_')
        df['tl3n'] = flawful.columns_with_prefix_to_list(df, 'tl3n_')
        ret_val = [ flawful.combine_answer_lists(prompts=tl3p, answers_1=tl3t,
                    answers_2=tl3n, answer1_hint=tl_abbr,
                    answer2_hint=nl_abbr)
                for (tl3p, tl3t, tl3n) in df[['tl3p','tl3t','tl3n']].values
                  ]
        #df['tl3_prompts_list'] = [ x['prompts'] for x in ret_val ]
        #df['tl3_list'] = [ x['answers'] for x in ret_val ]
        df['tl3_prompts_list'] = [ x['prompts'] for x in ret_val ]
        df['tl3_list'] = [ x['answers'] for x in ret_val ]
    elif ('tl3_prompts_list' in df or 'tl3_list' in df):
        raise ValueError('Both or none of `tl3_prompts_list` and `tl3_list`'
                         'columns should be present in `df`.')
    else:
        df['tl3_list'] = df.tl3_list.map(lambda x: [val.strip() for val in x])
        df['tl3_prompts_list'] = df.tl3_prompts_list.map(
                                         lambda x: [val.strip() for val in x])

    def braces_to_class_list(x):
        return [ flawful.braces_to_class(val, html_class=braces_html_class)
                 for val in x ]
    if braces_html_class is not None:
        df['tl3_prompts_list'] = df.tl3_prompts_list.map(braces_to_class_list)
        df['tl3_list'] = df.tl3_list.map(braces_to_class_list)

    ret_mtp = [
         flawful.make_hint_target_and_answer(
                     answer1=tl_answer, answer2=nl_answer,
                     answer1_list=tl_answer_list, answer2_list=nl_answer_list,
                     answer1_hint=tl_abbr,  answer2_hint=nl_abbr)
         for (tl_answer, nl_answer, tl_answer_list, nl_answer_list)
              in df[['tl_answer','nl_answer','tl_answer_list',
                     'nl_answer_list']].values
              ]
    df['target'] = [ x['hint'] + ': ' + x['target'] for x in ret_mtp ]
    df['answer'] = [ x['answer'] for x in ret_mtp ]
    df['tl_for_headword'] = np.where(df.tl_headword != '',
                                     df.tl_headword, df.tl1)
    df['merge_id'] = df.tl_for_headword.map(str_to_wordlist_key)
    res_mc = df.chaplist.apply(flawful.init_chapter,
                               str_to_chapter=str_to_chapter)
    df['min_chap'] = [ x['chapter'] for x in res_mc ]
    df['chap_tags'] = [ x['tags'] for x in res_mc ]

    res_tl1 = [
          flawful.tag_audio_and_markup(audio_dicts=aud_dicts,
                 wordlists=wordlists,
                 str_to_wordlist_key=str_to_wordlist_key,
                 str_to_audio_key=str_to_audio_key,
                 select_keys_no_audio=select_keys_no_audio,
                 htag_prefix=htag_prefix,
                 chapter=row[1],
                 tokens=[[row[0]]],
                 names=['tl1'],
                 assign_chapter=[True])
          for row in df[['tl1','min_chap']].values
              ]
    df['tl1_color'] = [''.join(x.markup_output['tl1']) for x in res_tl1]
    df['audio'] = [x.audio_output for x in res_tl1]
    df['chapter'] = [x.chapter for x in res_tl1]
    df['Tags'] = [x.tags for x in res_tl1]
    df['Tags'] = df['chap_tags'] + ' ' + df['Tags']
    df['prompt'] = (df.tl1_color + ' (' + df.part_of_speech + ') '
                       + df.target)
    df['prompt'] = np.where(df.tl1_hint == '', df.prompt,
                            df.prompt + ' [' + df.tl1_hint + ']')

    make_rv = [
        flawful.make_prompt_and_answer_table(
            prompts=[r[0],''], answers=[r[1],''],
            expr_prompts=r[2], expr_answers=r[3],
            drop_empty_rows=True)
       for r in df[['prompt','answer','tl3_prompts_list', 'tl3_list']].values
              ]
    df['tl_table_prompt'] = [ x['prompt'] for x in make_rv ]

    # rerun to make answer, only difference is `prompts` parameter
    make_rv = [
        flawful.make_prompt_and_answer_table(
            prompts=[r[0],r[4]], answers=[r[1],''],
            expr_prompts=r[2], expr_answers=r[3],
            drop_empty_rows=True)
       for r in df[['prompt','answer','tl3_prompts_list',
                    'tl3_list','tl2']].values
              ]
    df['tl_table_answer'] = [ x['answer'] for x in make_rv ]

    df = df[['id','merge_id','tl_table_answer', 'tl_table_prompt','pronun',
       'notes','audio','chapter','Tags']]

    return df

def create_tl_additional_output(df, aud_dicts, wordlists,
                 str_to_wordlist_key,
                 str_to_audio_key,
                 select_keys_no_audio,
                 flags,
                 tl_override_df = None,
                 str_to_chapter = None,
                 braces_html_class = None,
                 nl_abbr = 'N',
                 tl_abbr = 'T',
                 htag_prefix = 'TL',
                 flag_id_prefix = 'HW_',
                 output_mapper = None,
                              ) -> pd.DataFrame:
    """Create output file for `TL additional` notes.

    The `tl1` field is parsed and (1) tokens marked with '°' are
    identified.  (2) tokens NOT marked with '°' or '†' are then identified.
    A row in the output will be created for each headword in (1) that is
    not in the set of headwords for (2).

    See module docstring for background.

    A given headword will not be duplicated in the output.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with the following columns:
        - nl1 : Native language word or phrase
        - tl1_list : Primary target language word or phrase(s)
        - tl2_list : Secondary target language word or phrase(s).
        - tl_notes_list : Notes, including definitions. This is a
                list of tokens. If a token matches the format in
                `to_note_dict`, it will be processed and definition
                information extracted. Other tokens are ignored.
        - tl_pronun : Contains pronunciation information. This is passed
                through to the output file.
        - part_of_speech : Part of speech. Used to generate prompt.
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
    tl_override_df : pd.DataFrame, optional
        Additional input data frame for generating notes / cards where the
        target language word is on the front. This will override any cards
        with the same headword generated from `df`. This is done because
        the cards generated from `df` have some limitations. The front side
        of the card is basically just what is extracted from `tl1`, and
        there was no easy way to add hints or a table of other prompts for
        expressions containing the headword.

        The required columns (in any order) are:
        - id : A unique number or string for each input row
        - tl1 : Similar meaning as main file, except this should be a
                single word or phrase and not a token-delimited list.
        - tl_headword : If populated, will generate the headword for merging to
                the flagged records. Otherwise, `tl1` will be used.
        - tl2 : Similar meaning as on main file, except this should be a
                single word or phrase and not a token-delimited list.
        - tl_answer : The meaning(s) of `tl1` in the target language, if
                available. This is a string.
        - tl_answer_list : `tl_answer`, but tokenized into a list. This is
                used currently to determine how many answers are expected,
                which is prinsted in the prompt.
        - nl_answer : The meaning(s) of `tl1` in the native language. At
                least one of `tl_answer` or `nl_answer` must be populated.
        - nl_answer_list : `nl_answer`, but tokenized into a list. Same
                rationale as for `tl_answer_list`.
        - notes : Same meaning as `tl_notes` in the primary input file,
                except this field is not parsed for synonyms. It is only
                passed through to the output.
        - pronun : Same meaning as `tl_pronun` in the primary input file
        - tl1_hint : Same meaning as in the primary input file.
        - part_of_speech : Same meaning as in the primary input file.
        - chaplist : Same meaning as in the primary input file.

        In addition, fields can exist that are analagous to `tl3` and
        `tl3_prompts` in the main file. These should either be of the form:
           - tl3_list : Same meaning as `tl3` in the main file, except
             is a list of the tokens instead of a string.
           - tl3_prompts_list : Same meaning as `tl3_prompts` in the main
             file, except is a list of the tokens instead of a string.
        Alternatively, users can use fields that will be used to generate
        `tl3_list` and `tl3_prompts_list`. In this case, the input fields
        should each contain a single token.
           - tl3p_N : The Nth token for `tl3_prompts_list`
           - tl3t_N : The Nth token for `tl3_list` (in target language)
           - tl3n_N : The Nth token for `tl3_list` (in native language)
    str_to_chapter : Callable[[str], int], optional
        Function to convert strings in `tl_override_df.chaplist` to integer
        representing the minimum chapter. Passed to `init_chapter()`. See
        that function for details. Must be populated if `tl_override_df` is
        not None.
    braces_html_class : str, optional
        If populated, when processing `tl_override_df` (if applicable),
        '{text}' in the `tl3_list` or `tl3_prompts_list` tokens is
        converted to: '<div class={braces_html_class}>text</div>'.
    output_mapper : optional
        If not `None`, `pd.DataFrame.rename` will be called on the dataset
        that makes the additional output file just before the output is
        written, and this will be passed as the mapper.
    nl_abbr : str, optional (default='N')
        The additional file could have answers in `tl_answer`, `nl_answer`
        or both. The card types derived from flagged words in `tl1` could
        have an answer also from `tl1`, `nl1`, or `tl_notes_list`. This
        hint is added to the prompt created on the front of the card
        indicating from which field the answer was taken.
    tl_abbr : str, optional (default='T')
        See `nl_abbr` above.
    htag_prefix : str, optional (default='TL')
        Eventually passed to `flag_audio_and_markup`.
    flag_id_prefix : str, optional (default='HW_')
        Prefix to add to the front of the headword when creating `note_id`
        for records created from flagged `tl1` words.

    Returns
    -------
    pd.DataFrame. Also, `aud_dicts` and `wordlists` are updated as
    side-effects.
    """

    tl1_not_flagged_set = set()
    tl1_flagged_dict = {}

    df.tl1_list.map(lambda x: _add_unflagged_headword_to_set(x,
          tl1_not_flagged_set, str_to_wordlist_key, flags))

    for  (nl1,    part_of_speech, tl_notes_list, tl_pronun,
          tl1_list,    tl2_list) in df[
        ['nl1', 'part_of_speech','tl_notes_list','tl_pronun',
         'tl1_list', 'tl2_list']].values:
        _make_new_cards(exclude_headwords=tl1_not_flagged_set,
                       tl1_flagged_dict_=tl1_flagged_dict,
                       str_to_wordlist_key=str_to_wordlist_key,
                       flags=flags,
                       nl1_hint=nl_abbr, tl1_hint=tl_abbr,
                       nl1=nl1, part_of_speech=part_of_speech,
                       tl1_list=tl1_list, tl2_list=tl2_list,
                       tl_notes_list=tl_notes_list, tl_pronun=tl_pronun)
    tl1_df = pd.DataFrame.from_dict(tl1_flagged_dict, orient='index')

    res_tl1 = [
          flawful.tag_audio_and_markup(audio_dicts=aud_dicts,
                 wordlists=wordlists,
                 str_to_wordlist_key=str_to_wordlist_key,
                 str_to_audio_key=str_to_audio_key,
                 select_keys_no_audio=select_keys_no_audio,
                 htag_prefix=htag_prefix,
                 chapter=999,
                 tokens=[[row[0]]],
                 names=['tl1'],
                 assign_chapter=[True])
          for row in tl1_df[['tl1']].values
              ]
    tl1_df['tl_audio'] = [x.audio_output for x in res_tl1]
    tl1_df['tl1_color'] = [''.join(x.markup_output['tl1']) for x in res_tl1]
    tl1_df['chapter'] = [x.chapter for x in res_tl1]
    tl1_df['Tags'] = [x.tags for x in res_tl1]
    vars_in_output = ['note_id', 'nl1', 'part_of_speech', 'tl_defs',
                     'answer_lang', 'tl1', 'tl2', 'tl_pronun', 'tl_audio',
                     'tl1_color', 'chapter']

    if tl_override_df is not None:
        df2 = _process_tl_override_df(df=tl_override_df,
                     str_to_chapter=str_to_chapter,
                     aud_dicts=aud_dicts, wordlists=wordlists,
                     str_to_wordlist_key=str_to_wordlist_key,
                     str_to_audio_key=str_to_audio_key,
                     braces_html_class=braces_html_class,
                     select_keys_no_audio=select_keys_no_audio,
                     nl_abbr=nl_abbr, tl_abbr=tl_abbr,
                     htag_prefix=htag_prefix,
                     )
        df2 = df2.rename({'audio': 'o_audio', 'chapter': 'o_chapter',
                    'Tags': 'o_Tags'}, axis='columns')

        tl1_df = tl1_df.merge(df2[['id','merge_id','tl_table_answer',
           'tl_table_prompt','pronun','notes','o_audio','o_chapter','o_Tags']],
            how='outer', on='merge_id', indicator='merge_ind')

        in_df2 = tl1_df.merge_ind != 'left_only'
        tl1_df['has_table'] = np.where(in_df2, 'has_table', '')
        tl1_df['no_table']  = np.where(in_df2, '', 'Y')
        tl1_df['tl_audio'] = np.where(in_df2, tl1_df.o_audio,  tl1_df.tl_audio)
        tl1_df['chapter']  = np.where(in_df2, tl1_df.o_chapter, tl1_df.chapter)
        tl1_df['Tags']     = np.where(in_df2, tl1_df.o_Tags,   tl1_df.Tags)
        tl1_df['note_id']  = np.where(in_df2, tl1_df.id,
                                      flag_id_prefix + tl1_df.merge_id)
        vars_in_output.extend(['tl_table_answer','tl_table_prompt','has_table',
                               'no_table','pronun','notes'])

    #tl1_df['dummy'] = True
    #print(flawful.twowaytbl(tl1_df, 'chapter', 'dummy', cumulative=True))
    tl1_df = tl1_df[vars_in_output + ['Tags']]
    return tl1_df

    #if output_mapper is not None:
    #    tl1_df.rename(columns=output_mapper, inplace=True)
    #
    #tl1_df.to_csv(f'{outfile}.txt', sep='\t', quoting=csv.QUOTE_NONE,
    #              index=False)
    #tl1_df[0:0].to_csv(f'{outfile}_fields.txt', sep='\t',
    #                   quoting=csv.QUOTE_NONE, index=False)

    #--------------------------------------------------------------------------
    # Alternate output that adds metadata to the file header
    #--------------------------------------------------------------------------
    # The code that writes to '{outfile}.txt' and f'{outfile}_fields.txt' above
    # can be replaced with the below to add metadata to the file header. See
    # similar section in example1.py for discussion and cautions.
    #
    #tags_col = tl1_df.columns.get_loc('Tags')
    #column_str = '\t'.join(tl1_df.columns)
    #metadata = ['#separator:Tab',
    #           f'#deck:TL Additional',
    #            "#notetype:TL Additional",
    #            '#html:true',
    #           f'#tags column:{tags_col + 1}',
    #           f"#if matches:update current",
    #           f"#columns:{column_str}",
    #           ]
    #meta_df = pd.DataFrame(metadata)
    #meta_df.to_csv(f'{outfile}.txt', sep='@', index=False, header=False,
    #             quoting=csv.QUOTE_NONE)
    #tl1_df.to_csv(f'{outfile}.txt', sep='\t', quoting=csv.QUOTE_NONE,
    #              index=False, mode='a')
