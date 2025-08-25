"""Test sentence-based chunking with customizable delimiters."""

import pytest
from yourbench.utils.chunking_utils import split_into_sentences, split_into_sentence_chunks


class TestSentenceSplitting:
    """Test sentence splitting with various delimiters."""
    
    def test_english_sentence_splitting(self):
        """Test splitting English text with default delimiters."""
        text = "This is a sentence. This is another! Is this a question?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is a sentence."
        assert sentences[1] == "This is another!"
        assert sentences[2] == "Is this a question?"
    
    def test_chinese_sentence_splitting(self):
        """Test splitting Chinese text with Chinese delimiters."""
        text = "这是第一句话。这是第二句话！这是问句吗？"
        sentences = split_into_sentences(text, delimiters=r"[。！？]")
        assert len(sentences) == 3
        assert sentences[0] == "这是第一句话。"
        assert sentences[1] == "这是第二句话！"
        assert sentences[2] == "这是问句吗？"
    
    def test_mixed_language_splitting(self):
        """Test splitting mixed Chinese-English text."""
        text = "Hello world. 你好世界。How are you? 你好吗？"
        sentences = split_into_sentences(text, delimiters=r"[.?。？]")
        assert len(sentences) == 4
        assert "Hello world." in sentences[0]
        assert "你好世界。" in sentences[1]
    
    def test_empty_text(self):
        """Test handling of empty text."""
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []
    
    def test_text_without_delimiters(self):
        """Test text without sentence delimiters."""
        text = "This text has no sentence delimiters"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text


class TestSentenceChunking:
    """Test sentence-based chunking functionality."""
    
    def test_basic_chunking(self):
        """Test basic sentence chunking."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = split_into_sentence_chunks(text, max_sentences=2, overlap_sentences=0, min_chunk_length=10)
        assert len(chunks) == 3
        assert "Sentence one. Sentence two." in chunks[0]
        assert "Sentence three. Sentence four." in chunks[1]
        assert "Sentence five." in chunks[2]
    
    def test_chunking_with_overlap(self):
        """Test sentence chunking with overlap."""
        text = "S1. S2. S3. S4. S5."
        chunks = split_into_sentence_chunks(text, max_sentences=3, overlap_sentences=1, min_chunk_length=5)
        assert len(chunks) == 2
        # First chunk: S1, S2, S3
        # Second chunk: S3, S4, S5 (overlaps with S3)
        assert "S3." in chunks[0] and "S3." in chunks[1]
    
    def test_chinese_chunking(self):
        """Test chunking Chinese text."""
        text = "第一句。第二句。第三句。第四句。第五句。"
        chunks = split_into_sentence_chunks(
            text, 
            max_sentences=2, 
            overlap_sentences=0,
            delimiters=r"[。]",
            min_chunk_length=1
        )
        assert len(chunks) == 3
        assert "第一句。" in chunks[0]
        assert "第三句。" in chunks[1]
        assert "第五句。" in chunks[2]
    
    def test_min_chunk_length(self):
        """Test minimum chunk length enforcement."""
        text = "Short. Also short. This is a much longer sentence that exceeds minimum."
        chunks = split_into_sentence_chunks(
            text,
            max_sentences=1,
            overlap_sentences=0,
            min_chunk_length=50
        )
        # Short chunks should be merged
        assert len(chunks) <= 2
    
    def test_single_sentence(self):
        """Test chunking with a single sentence."""
        text = "This is just one sentence."
        chunks = split_into_sentence_chunks(text, max_sentences=5)
        assert len(chunks) == 1
        assert chunks[0] == text