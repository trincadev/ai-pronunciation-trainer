from tests import EVENTS_FOLDER

text_dict = {"de": "Hallo, wie geht es dir?", "en": "Hi there, how are you?"}
real_transcript = {
    "de": "ich bin alexwe beste du",
    "en": "hi how are you"
}
real_transcripts_ipa = {
    "de": "haloː, viː ɡeːt ɛːs diːr?",
    "en": "haɪ ðɛr, haʊ ər ju?"
}
expected_GetAccuracyFromRecordedAudio = {
    "de": {
        "real_transcript": "ich bin om werbst du wille freude wo no wie essen",
        "ipa_transcript": "ɪç biːn oːm vɛːrbst duː vɪlɛː frɔɪ̯dɛː voː noː viː ɛzɛːn",
        "pronunciation_accuracy": 62.0,
        "real_transcripts": "Ich bin Tom, wer bist du? Viel Freude. Wollen wir essen?",
        "matched_transcripts": "ich bin om - - du wille freude wo wie essen",
        "real_transcripts_ipa": "ɪç biːn toːm, vɐ bɪst duː? fiːl frɔɪ̯dɛː. vɔln̩ viːɐ̯ ɛzɛːn?",
        "matched_transcripts_ipa": "ɪç biːn oːm - - duː vɪlə frɔɪ̯də voː viː ɛzɛːn",
        "pair_accuracy_category": "0 0 1 2 2 0 2 0 2 1 0",
        "start_time": "0.625875 0.8644375 1.3415625 5.7945625 5.7945625 2.772875 3.4885 3.886125 4.919875 5.51625 5.7945625",
        "end_time": "0.9644375 1.203 1.6800625 6.371625 6.371625 3.1114375 3.986125 4.46325 5.258375 5.815 6.371625",
        "is_letter_correct_all_words": "111 111 0111 000 0000 111 0101 1111111 110000 110 111111 ",
    },
    "en": {
        "real_transcript": "tom weing as someone else ca",
        "ipa_transcript": "tɑm weing ɛz ˈsəmˌwən ɛls ˈsiˈeɪ",
        "pronunciation_accuracy": 75.0,
        "real_transcripts": "Tom is wearing someone else's coat.",
        "matched_transcripts": "tom - weing someone else ca",
        "real_transcripts_ipa": "tɑm ɪz ˈwɛrɪŋ ˈsəmˌwən ˈɛlsɪz koʊt.",
        "matched_transcripts_ipa": "tɑm  weing ˈsəmˌwən ɛls ˈsiˈeɪ",
        "pair_accuracy_category": "0 2 0 0 0 2",
        "start_time": "1.4094375 3.4605 2.0405 2.671625 3.0660625 3.4605",
        "end_time": "1.903875 3.7971875 2.5744375 3.1660625 3.5605 3.7971875",
        "is_letter_correct_all_words": "111 00 1100111 1111111 111110 10101 ",
    },
}
expected_get_speech_to_score = {
    "de": {
        "real_transcript": real_transcript["de"],
        "ipa_transcript": "ɪç biːn aːlɛksvɛː bɛstɛː duː",
        "pronunciation_accuracy": 18.0,
        "real_transcripts": text_dict["de"],
        "matched_transcripts": "ich bin beste - du",
        "real_transcripts_ipa": real_transcripts_ipa["de"],
        "matched_transcripts_ipa": "ɪç biːn bəstə - duː",
        "pair_accuracy_category": "2 2 2 2 2",
        "start_time": "0.0 0.3075 1.5785625 2.1346875 2.1346875",
        "end_time": "0.328 0.6458125 2.15525 2.4730625 2.4730625",
        "is_letter_correct_all_words": "100001 010 0101 00 1001 ",
    },
    "en": {
        "real_transcript": real_transcript["en"],
        "ipa_transcript": "haɪ haʊ ər ju",
        "pronunciation_accuracy": 69.0,
        "real_transcripts": text_dict["en"],
        "matched_transcripts": "hi - how are you",
        "real_transcripts_ipa": real_transcripts_ipa["en"],
        "matched_transcripts_ipa": "haɪ  haʊ ər ju",
        "pair_accuracy_category": "0 2 0 0 0",
        "start_time": "0.2245625 1.3228125 0.852125 1.04825 1.3228125",
        "end_time": "0.559875 1.658125 1.14825 1.344375 1.658125",
        "is_letter_correct_all_words": "11 000001 111 111 1111 ",
    },
}
expected_with_audio_files_splitted_list = {
    "de": {
        "audio_files": [
            f'{EVENTS_FOLDER / "test_de__part0_start0.0_end0.328..wav"}',
            f'{EVENTS_FOLDER / "test_de__part1_start0.3075_end0.6458125..wav"}',
            f'{EVENTS_FOLDER / "test_de__part2_start1.5785625_end2.15525..wav"}',
            f'{EVENTS_FOLDER / "test_de__part3_start2.1346875_end2.4730625..wav"}',
            f'{EVENTS_FOLDER / "test_de__part4_start2.1346875_end2.4730625..wav"}',
        ],
        "audio_durations": [
            0.328,
            0.3383125,
            0.5766875,
            0.3383750000000001,
            0.3383750000000001,
        ],
        "real_transcript": real_transcript["de"],
        "ipa_transcript": "ɪç biːn aːlɛksvɛː bɛstɛː duː",
        "pronunciation_accuracy": 18.0,
        "real_transcripts": text_dict["de"],
        "matched_transcripts": "ich bin beste - du",
        "real_transcripts_ipa": real_transcripts_ipa["de"],
        "matched_transcripts_ipa": "ɪç biːn bəstə - duː",
        "pair_accuracy_category": "2 2 2 2 2",
        "start_time": "0.0 0.3075 1.5785625 2.1346875 2.1346875",
        "end_time": "0.328 0.6458125 2.15525 2.4730625 2.4730625",
        "is_letter_correct_all_words": "100001 010 0101 00 1001 ",
    },
    "en": {
        "audio_files": [
            f'{EVENTS_FOLDER / "test_en__part0_start0.2245625_end0.559875..wav"}',
            f'{EVENTS_FOLDER / "test_en__part1_start1.3228125_end1.658125..wav"}',
            f'{EVENTS_FOLDER / "test_en__part2_start0.852125_end1.14825..wav"}',
            f'{EVENTS_FOLDER / "test_en__part3_start1.04825_end1.344375..wav"}',
            f'{EVENTS_FOLDER / "test_en__part4_start1.3228125_end1.658125..wav"}',
        ],
        "audio_durations": [
            0.3353125,
            0.3353125000000001,
            0.29612499999999997,
            0.2961250000000002,
            0.3353125000000001,
        ],
        "real_transcript": real_transcript["en"],
        "ipa_transcript": "haɪ haʊ ər ju",
        "pronunciation_accuracy": 69.0,
        "real_transcripts": text_dict["en"],
        "matched_transcripts": "hi - how are you",
        "real_transcripts_ipa": real_transcripts_ipa["en"],
        "matched_transcripts_ipa": "haɪ  haʊ ər ju",
        "pair_accuracy_category": "0 2 0 0 0",
        "start_time": "0.2245625 1.3228125 0.852125 1.04825 1.3228125",
        "end_time": "0.559875 1.658125 1.14825 1.344375 1.658125",
        "is_letter_correct_all_words": "11 000001 111 111 1111 ",
    },
}
expected_with_selected_word_valid_index = {
    "de": {
        "audio_files": [
            f'{EVENTS_FOLDER / "test_de_easy__part0_start0.0_end0.4733125..wav"}',
            f'{EVENTS_FOLDER / "test_de_easy__part1_start0.3733125_end0.70425..wav"}',
            f'{EVENTS_FOLDER / "test_de_easy__part2_start0.60425_end0.8966875..wav"}',
            f'{EVENTS_FOLDER / "test_de_easy__part3_start0.7966875_end1.089125..wav"}',
            f'{EVENTS_FOLDER / "test_de_easy__part4_start0.989125_end1.3200625..wav"}',
        ],
        "audio_durations": [
            0.4733125,
            0.33093750000000005,
            0.2924375,
            0.2924374999999999,
            0.3309374999999999,
        ],
        "real_transcript": "hallo wie geht es dir",
        "ipa_transcript": "haloː viː ɡeːt ɛːs diːɐ̯",
        "pronunciation_accuracy": 100.0,
        "real_transcripts": text_dict["de"],
        "matched_transcripts": "hallo wie geht es dir",
        "real_transcripts_ipa": real_transcripts_ipa["de"],
        "matched_transcripts_ipa": "haloː viː ɡeːt ɛːs diːɐ̯",
        "pair_accuracy_category": "0 0 0 0 0",
        "start_time": "0.0 0.3733125 0.60425 0.7966875 0.989125",
        "end_time": "0.4733125 0.70425 0.8966875 1.089125 1.3200625",
        "is_letter_correct_all_words": "111111 111 1111 11 1111 ",
    },
    "en": {
        "audio_files": [
            f'{EVENTS_FOLDER / "test_en_easy__part0_start0.0_end0.1625..wav"}',
            f'{EVENTS_FOLDER / "test_en_easy__part1_start0.0625_end0.3875..wav"}',
            f'{EVENTS_FOLDER / "test_en_easy__part2_start0.2875_end0.575..wav"}',
            f'{EVENTS_FOLDER / "test_en_easy__part3_start0.475_end0.8..wav"}',
            f'{EVENTS_FOLDER / "test_en_easy__part4_start0.7_end0.9875..wav"}',
        ],
        "audio_durations": [
            0.1625,
            0.325,
            0.2875,
            0.32500000000000007,
            0.2875000000000001,
        ],
        "real_transcript": "i there how are you",
        "ipa_transcript": "aɪ ðɛr haʊ ər ju",
        "pronunciation_accuracy": 94.0,
        "real_transcripts": text_dict["en"],
        "matched_transcripts": "i there how are you",
        "real_transcripts_ipa": real_transcripts_ipa["en"],
        "matched_transcripts_ipa": "aɪ ðɛr haʊ ər ju",
        "pair_accuracy_category": "2 0 0 0 0",
        "start_time": "0.0 0.0625 0.2875 0.475 0.7",
        "end_time": "0.1625 0.3875 0.575 0.8 0.9875",
        "is_letter_correct_all_words": "01 111111 111 111 1111 ",
    },
}