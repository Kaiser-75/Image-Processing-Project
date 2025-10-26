## Project 4 
## Title: Automated Playing Card Recognition using Template-Based Computer Vision
## Authors: Anna†, Kaiser†  
(† equal contribution)
## Algorithm Overview:
The algorithm detects a playing card by locating its contour against a plain background and warping it to a fixed upright view. It then crops the top-left corner, where the rank and suit symbols appear. The cropped region is binarized using an adaptive threshold and split into two parts, one for rank and one for suit. Each part is cleaned, resized to a standard size, and compared to pre-made templates using normalized cross-correlation. The template with the highest similarity is selected as the card’s rank or suit. This process makes the system invariant to rotation, scale, and lighting.

## Directory Overview



# Document link: [PDF File](https://drive.google.com/file/d/17xpe8O7JdY2evqKMsyXgTrJEmiOorkPA/view?usp=drive_link)


