import os
import asyncio
from playwright.async_api import async_playwright

class ReverseImageSearch:
    """
    Perform reverse image search using Google Images and save a screenshot of the results.
    """

    def __init__(self):
        pass

    def search_and_screenshot(self, image_path: str, screenshot_path: str, delay: float = 2., headless: bool = True):
        """
        Perform reverse image search and save screenshot.
        Inputs:
        - image_path: (str) Path to the image to query,
        - screenshot_path: (str) Path to save the screenshot.
        - delay: (float) Delay in seconds to wait for the search results to load.
        - headless: (bool) Whether to run the browser in headless mode.
        """
        try:
            screenshot_file = self._run_search_by_image(image_path, screenshot_path, delay, headless)
            print(f'Screenshot saved at {screenshot_file}')
        except Exception as e:
            print(e)

    def _run_search_by_image(self, image_path: str, screenshot_path: str, delay: float = 2., headless=False):
        """ Run Playwright-based image search and return screenshot """
        try:
            return asyncio.run(search_by_image(image_path, screenshot_path, delay, headless))
        except Exception as e:
            print(f'Error in reverse_image_search: {e}')
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_by_image(image_path, screenshot_path, delay, headless))
    


async def search_by_image(image_path, screenshot_path='search_results.png', delay=8., headless=False):
    """
    Perform a reverse image search using Playwright library with a local image file.
    Inputs:
    - image_path: (str) Path to the image file to search for,
    - screenshot_path: (str) Path to save the screenshot.
    - delay: (float) Delay in seconds to wait for the search results to load.
    - headless: (bool) Whether the browser should run in headless mode.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()
        # print('1')

        await page.goto('https://images.google.com')

        # Click the "Search by image" button
        await page.click('div[aria-label="Search by image"]')

        # Click the "Upload an image" tab
        await page.click('span[role="button"][jsname="tAPGc"]')

        # Upload the image file
        input_file = await page.query_selector('input[type="file"]')
        await input_file.set_input_files(image_path)

        # Wait for the search results to load
        await page.wait_for_selector('img', state='visible')
        await asyncio.sleep(delay)

        # Take a screenshot of the search results
        await page.screenshot(path=screenshot_path, full_page=True)

        await browser.close()

    return screenshot_path

def reverse_search_for_all_images_in_folder(folder_path: str, screenshot_folder: str, delay: float = 1., headless: bool = True):
    """
    Perform reverse image search for all images in a folder and save screenshots.
    Inputs:
    - folder_path: (str) Path to the folder containing images.
    - screenshot_folder: (str) Path to the folder where screenshots will be saved.
    - delay: (float) Delay in seconds to wait for the search results to load.
    - headless: (bool) Whether to run the browser in headless mode.
    """
    # Create the screenshot folder if it doesn't exist
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # Create an instance of ReverseImageSearch
    reverse_image_search = ReverseImageSearch()

    # Iterate over all images in the folder
    for image_filename in os.listdir(folder_path):
        if image_filename.lower().endswith(('.jpeg', '.jpg', '.png')):  # Check for image file types
            image_path = os.path.join(folder_path, image_filename)

            # Create the screenshot path by appending '-search_result.png' to the original filename
            base_name, _ = os.path.splitext(image_filename)

            screenshot_path = os.path.join(screenshot_folder, f"{base_name}-search_result.png")
            if os.path.exists(screenshot_path):
                print(screenshot_path, 'exists')
                continue
            # Perform reverse image search and save the screenshot
            reverse_image_search.search_and_screenshot(image_path, screenshot_path, delay, headless)

if __name__ == "__main__":
    # Path to the folder containing the images
    folder_path = "/workspace/RAG/reverse-image-rag-main/image_okvqa"

    # Path to the folder where the screenshots will be saved
    screenshot_folder = "/workspace/RAG/reverse-image-rag-main/screenshot"

    # Perform reverse image search for all images in the folder
    reverse_search_for_all_images_in_folder(folder_path, screenshot_folder, delay=3)
