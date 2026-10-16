from pptx import Presentation
from pptx.util import Inches
import datetime

# Load the existing presentation
prs = Presentation('/Users/shameel/Desktop/qwen_agent_test/Science_News_Presentation/Science_News_This_Week.pptx')

# Add a new slide for each article found
for i, article in enumerate(['Article 1', 'Article 2', 'Article 3']):
    slide_layout = prs.slide_layouts[5]  # Use a blank layout
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    if len(slide.placeholders) > 1:
        subtitle = slide.placeholders[1]
        subtitle.text = article
    else:
        body = slide.shapes.placeholders[0]
        body.text = f'Article {i+1} Title\n{article}

# Save the updated presentation
date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
new_filename = f'Science_News_This_Week_Updated_{date_str}.pptx'
prs.save(new_filename)
