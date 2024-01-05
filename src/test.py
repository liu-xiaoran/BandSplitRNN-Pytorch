import cydifflib

# 原始歌词
original_lyrics = "这里是原始歌词"
# 识别出的歌词
recognized_lyrics = "这里是识别歌词"

# 创建 SequenceMatcher 对象
matcher = cydifflib.SequenceMatcher(None, original_lyrics, recognized_lyrics)

# 找出匹配的部分
matches = matcher.get_matching_blocks()

# 打印匹配结果
for match in matches:
    start_orig = match.a
    start_recog = match.b
    length = match.size
    matching_text = original_lyrics[start_orig:start_orig+length]
    print(f"原始歌词位置: {start_orig}, 识别歌词位置: {start_recog}, 长度: {length}, 匹配内容: '{matching_text}'")


# 1 demucs
# 2 ts+word
# 3 转读音
# 4 align：匈牙利算法

# https://github.com/sanchit-gandhi/whisper-jax

# 伴奏分离：https://github.com/amanteur/BandSplitRNN-Pytorch

# 镜像：https://hf-mirror.com/

# https://github.com/facebookresearch/demucs

# https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file

# root@192.168.0.31