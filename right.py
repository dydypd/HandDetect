from playwright.sync_api import sync_playwright
import time

# Thông tin đăng nhập
EMAIL = "mi288nh@gmail.com"  # Thay thế bằng email của bạn
PASSWORD = "m1234567@"      # Thay thế bằng mật khẩu của bạn

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
    )

    context = browser.new_context(
        viewport=None,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        locale="en-US"
    )

    # Patch để ẩn navigator.webdriver
    context.add_init_script("""Object.defineProperty(navigator, 'webdriver', {get: () => undefined})""")

    page = context.new_page()
    page.goto("https://accounts.google.com/")

    # Tự động đăng nhập
    print("🔄 Đang tự động đăng nhập...")
    time.sleep(2)  # Đợi thêm để chắc chắn form đã load

    # Nhập email
    page.fill('input[type="email"]', EMAIL)
    page.click('button:has-text("Next")')
    time.sleep(2)  # Đợi thêm để chắc chắn form đã load

    # Đợi để form mật khẩu hiện ra
    page.wait_for_selector('input[type="password"]', state='visible')
    time.sleep(2)  # Đợi thêm để chắc chắn form đã load
    
    # Nhập mật khẩu
    page.fill('input[type="password"]', PASSWORD)
    page.click('button:has-text("Next")')
    
    # Đợi cho quá trình đăng nhập hoàn tất
    # Có thể cần điều chỉnh selector này tùy theo trang đích sau khi đăng nhập
    page.wait_for_load_state('networkidle')
    time.sleep(5)  # Đợi thêm để đảm bảo đăng nhập hoàn tất
    
    print("✅ Đăng nhập thành công!")

    # Lưu session sau khi đăng nhập thành công
    context.storage_state(path="google_session.json")
    print("✅ Đã lưu session vào google_session.json")

    browser.close()