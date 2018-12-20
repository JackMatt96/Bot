print('Start')
import logging
import os
import cv2
import numpy as np

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


def start(bot, update):
    update.effective_message.reply_text("Hi!")

def start_merge(bot, update):
    update.effective_message.reply_text("Starting merging session!")
    dp.add_handler(MessageHandler(Filters.photo, merge_photo))

def stop_merge(bot, update):
    update.effective_message.reply_text("Stopping merging session!")
    dp.remove_handler(merge_photo)

def replay(bot, update):
    update.effective_message.reply_text("Hai detto " +update.effective_message.text)

def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)

def merge_photo(bot, update):
    File = bot.get_file(update.effective_message.photo[-1].file_id)
    if(id[0]==None):
        id[0] = 'Temp/'+str(File.id)+'.jpg'
        update.effective_message.reply_text(id[0])
        File.download(id[0])
    else:
        id[1] = 'Temp/'+str(File.id)+'.jpg'
        update.effective_message.reply_text(id[1])
        File.download(id[1])

        match, toSend = stitching_images(cv2.imread(id[0]), cv2.imread(id[1]))
        nameFile = 'Temp/' + str(update.effective_chat.id) + '.jpg'
        nameFileMatch = 'Temp/' + str(update.effective_chat.id) + '_Match.jpg'
        cv2.imwrite(nameFile, toSend)
        cv2.imwrite(nameFileMatch, toSendMatch)
        update.effective_message.send_Photo(photo = open(nameFile,'rb'))
        update.effective_message.send_Photo(photo = open(nameFileMatch,'rb'))
        id = (None, None)


MAX_FEATURES = 500
K = 4
def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
	return output_img



def stitching_images(image1, image2):
    img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, K)

    
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    match = np.asarray(good)
    
    if len(match[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in match[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in match[:,0] ]).reshape(-1,1,2)

    imMatches = cv2.drawMatchesKnn(image1, src, image2, dst, matches, None)    

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    return imMatches, warpImages(image2, image1, H)


if __name__ == "__main__":
    # Set these variable to the appropriate values
    TOKEN = os.environ.get('TELEGRAM_TOKEN')
    NAME = "first-jack-bot"

    # Port is given by Heroku
    PORT = os.environ.get('PORT')

    # Enable logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up the Updater
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    # Add handlers
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('start_merge', start_merge))
    dp.add_handler(CommandHandler('stop_merge', stop_merge))
    
    dp.add_handler(MessageHandler(Filters.text, replay)) 
    dp.add_error_handler(error)
    id = (None, None)
    if not os.path.exists('Temp'):
        os.makedirs('Temp')

    # Start the webhook
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook("https://{}.herokuapp.com/{}".format(NAME, TOKEN))
    updater.idle()
    print('Ready')
