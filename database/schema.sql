CREATE TABLE crawler_pages (
    id SERIAL PRIMARY KEY,

    url TEXT UNIQUE NOT NULL,
    domain TEXT,

    raw_html TEXT,
    clean_text TEXT,

    status TEXT,
    status_code INT,

    depth INT DEFAULT 0,

    content_hash TEXT,

    language TEXT,

    error_message TEXT,

    crawl_time TIMESTAMP DEFAULT NOW()
);

CREATE TABLE url_metadata (
    id SERIAL PRIMARY KEY,

    url TEXT UNIQUE NOT NULL,

    status_code INT,
    content_type TEXT,
    response_size INT,

    discovered_from TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tokenizer_data (
    id SERIAL PRIMARY KEY,

    url TEXT,

    clean_text TEXT NOT NULL,

    language TEXT,

    token_count INT,

    content_hash TEXT UNIQUE
);

-- INDEXES
CREATE INDEX idx_crawler_domain
ON crawler_pages(domain);

CREATE INDEX idx_tokenizer_language
ON tokenizer_data(language);

CREATE INDEX idx_content_hash
ON crawler_pages(content_hash);