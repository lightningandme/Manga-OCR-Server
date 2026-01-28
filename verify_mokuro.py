import sys
import os
import shutil
import sqlite3
import json
from pathlib import Path

# Add project root to path
sys.path.append(r"c:\PycharmProjects\Manga-OCR-Server")

from suwayomigo_service import mokuro_processor

# Setup Test Env
TEST_MANGA_ID = 99999
TEST_CHAP = 1
TEST_DIR = mokuro_processor.STORAGE_ROOT / "TestManga" / str(TEST_MANGA_ID) / str(TEST_CHAP)

def cleanup():
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR.parent.parent)
        except:
            pass
    # Clean DB
    try:
        conn = sqlite3.connect(mokuro_processor.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM manga_lines WHERE manga_id=?", (TEST_MANGA_ID,))
        conn.commit()
        conn.close()
    except:
        pass

def test_download_skip():
    print("Testing Download Skip...")
    cleanup()
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Create script.txt
    (TEST_DIR / "script.txt").touch()
    
    # 2. Call download
    # Mock requests to fail if called
    original_get = mokuro_processor.requests.get
    def fail_request(*args, **kwargs):
        raise Exception("Should not be called")
    mokuro_processor.requests.get = fail_request
    
    try:
        status = mokuro_processor.download_single_page("url", "u", "p", "TestManga", TEST_MANGA_ID, TEST_CHAP, 1)
        assert status == 0
        print("PASS: Download skipped when script.txt exists.")
    finally:
        mokuro_processor.requests.get = original_get
    
    cleanup()

def test_generate_db_and_delete():
    print("Testing Generate Script Logic...")
    cleanup()
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create fake mokuro file - name must match dir name?
    # mokuro_processor logic: mokuro_filename = target_path.name + ".mokuro" 
    # target_path is checking if it ends with .mokuro or looking in parent.
    # target_dir is passed. logic: mokuro_filename = Path(target_dir).name + ".mokuro"
    # target_dir is `ch_idx` (e.g. "1")
    
    mokuro_name = TEST_DIR.name + ".mokuro"
    mokuro_file = TEST_DIR.parent / mokuro_name # Logic line 100: target_path.parent / filename
    
    # Make sure parent exists
    mokuro_file.parent.mkdir(parents=True, exist_ok=True)

    mokuro_data = {
        "pages": [
            {
                "img_width": 100, "img_height": 200,
                "blocks": [{"box": [0,0,10,10], "lines": ["Hello"]}]
            }
        ]
    }
    with open(mokuro_file, 'w', encoding='utf-8') as f:
        json.dump(mokuro_data, f)
        
    # Create fake image
    (TEST_DIR / "001.jpg").touch()
    
    # Call generate
    mokuro_processor.generate_script_file(TEST_DIR, "TestManga", TEST_MANGA_ID, TEST_CHAP)
    
    # Check 1: Image deleted
    if (TEST_DIR / "001.jpg").exists():
        print("FAIL: Image NOT deleted")
    else:
        print("PASS: Image deleted.")
    
    # Check 2: script.txt exists
    if (TEST_DIR / "script.txt").exists():
        print("PASS: script.txt generated.")
    else:
        print("FAIL: script.txt MISSING")
        
    # Check 3: DB has entry
    conn = sqlite3.connect(mokuro_processor.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT content FROM manga_lines WHERE manga_id=?", (TEST_MANGA_ID,))
    result = c.fetchone()
    conn.close()
    
    if result and result[0] == "Hello":
        print("PASS: Data in DB.")
    else:
        print(f"FAIL: DB Content Mismatch or Missing. Got: {result}")
    
    cleanup()

if __name__ == "__main__":
    try:
        test_download_skip()
        test_generate_db_and_delete()
        print("\nAll Tests Passed!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
