from wrapper import align, preprocess_from_file, write_csv

n_phone_class = 41
n_pitch_class = 47
phone_blank = 40
lang = 'english'
phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH',
          'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V',
          'W', 'Y', 'Z', 'ZH', ' ']

audio_file = "/home/ryan/PycharmProjects/jamendolyrics1/vocals/Cortez_-_Feel__Stripped__vocals.mp3"  # pre-computed source-separated vocals; These models do not work with mixture input.
lyrics_file = "/home/ryan/PycharmProjects/jamendolyrics1/lyrics/Cortez_-_Feel__Stripped_.raw.txt"  # example: jamendolyrics/lyrics/*.raw.txt"
word_file = "/home/ryan/PycharmProjects/jamendolyrics1/lyrics/Cortez_-_Feel__Stripped_.words.txt"  # example: jamendolyrics/lyrics/*.words.txt"; Set to None if you don't have it
method = 'Baseline'  # "Baseline", "MTL", "Baseline_BDR", "MTL_BDR"
checkpoint_path = '/home/ryan/PycharmProjects/LyricsAlignment-MTL1/checkpoints/checkpoint_Baseline'
cuda = True  # set True if you have access to a GPU

pred_file = f"./{method}_2044.csv"  # saved alignment results, "(float) start_time, (float) end_time, (string) word"

# load audio and lyrics
# words:        a list of words
# lyrics_p:     phoneme sequence of the target lyrics
# idx_word_p:   indices of word start in lyrics_p
# idx_line_p:   indices of line start in lyrics_p
audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(audio_file, lyrics_file, word_file, lang)

# compute alignment
# word_align:   a list of frame indices aligned to each word
# words:        a list of words
word_align, words = align(audio, words, lyrics_p, idx_word_p, idx_line_p, n_phone_class, n_pitch_class, phone_blank,
                          phones, method=method, cuda=cuda, checkpoint_path=checkpoint_path)

# write to csv
# can be imported to Sonic Visualiser
write_csv(pred_file, word_align, words)
