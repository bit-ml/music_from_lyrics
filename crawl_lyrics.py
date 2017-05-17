# Bitdefender 2017

"""
adapted from :
http://stackoverflow.com/questions/23904552/crawling-this-website-to-grab-lyrics
"""

import urllib
import re
import lxml.html
import unicodedata
import os
import sys


class LyricsCrawler(object):
    def __init__(self, artist, title):
        self.artist = self.__format_str(artist)
        self.title = self.__format_str(title)
        self.url = None
        self.lyric = None

    def __init__(self, artist, num_titles=None):
        self.artist = self.__format_str(artist)
        self.title = None
        self.url = None
        self.lyric = None
        self.max_num_titles = num_titles or 10

    def __format_str(self, s):
        # remove paranthesis and contents
        s = s.strip()
        try:
            # strip accent
            s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        except:
            pass
        s = s.title()
        return s

    def __quote(self, s):
        return urllib.parse.quote(s.replace(' ', '_'))

    def __make_url(self):
        artist = self.__quote(self.artist)
        title = self.__quote(self.title)
        artist_title = '%s:%s' %(artist, title)
        url = 'http://lyrics.wikia.com/' + artist_title
        self.url = url

    def update(self, artist=None, title=None):
        if artist:
            self.artist = self.__format_str(artist)
        if title:
            self.title = self.__format_str(title)


    def __grab_all_titles(self):
        artist = self.__quote(self.artist)
        url = 'http://lyrics.wikia.com/' + artist
        self.url = url
        doc = lxml.html.parse(self.url)
        lyricbox = doc.getroot().cssselect('.mw-content-ltr.mw-content-text')
        titles = []
        for node in lyricbox[0].getchildren():
            if (node.tag == "ol"):
                l = node.findall(".//a")
                for elm in l:
                    title = elm.get("title")
                    if ":" not in title:
                        continue
                    else:
                        title = title.split(":")[1]
                        titles += [title]
        return titles


    def lyricwikia(self):
        titles = self.__grab_all_titles()
        lyrics = []
        for t in titles[0:min(self.max_num_titles, len(titles))]:
            self.update(artist=None, title=t)
            self.__make_url()
            try:
                doc = lxml.html.parse(self.url)
                lyricbox = doc.getroot().cssselect('.lyricbox')[0]
            except IOError:
                continue
            lyric_set = []
            for node in lyricbox:
                if node.tag == 'br':
                    lyric_set.append('\n')
                if node.tail is not None:
                    lyric_set.append(node.tail)
            lyric =  "".join(lyric_set).strip()
            lyrics += [[t, lyric]]
        return lyrics


if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Need artist name!")
        sys.exit(-1)
    artist_name = " ".join(sys.argv[1:])
    artist_folder = os.path.join("lyrics", artist_name)
    if not os.path.exists(artist_folder):
        os.makedirs(artist_folder)

    crawler = LyricsCrawler(artist_name)
    lyrics = crawler.lyricwikia()
    for tl in lyrics:
        t = tl[0].split('/')[0]
        l = tl[1]
        f = open(os.path.join(artist_folder, t), "w")
        f.write(l)
        f.close()
