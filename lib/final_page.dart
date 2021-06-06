import 'package:adobe_xd/page_link.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:image_caption/home_page.dart';
import 'dart:io';

class FinalPage extends StatefulWidget {
  FinalPage({this.image});

  final File image;

  @override
  _FinalPageState createState() => _FinalPageState();
}

class _FinalPageState extends State<FinalPage> {
  final FirebaseFirestore _firebaseFirestore = FirebaseFirestore.instance;
  String caption = 'This is an image';

  Future<void> getCaption() async {
    var collection = await _firebaseFirestore.collection('predict').get();

    final List<DocumentSnapshot> documents = collection.docs;

    documents.forEach((element) {
      setState(() {
        caption = element.data()['predict'];
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFDBE9F6),
      body: Stack(
        children: [
          new Image.asset("assets/images/finalpage.jpg"),
          Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Container(
                  child: Image.file(widget.image),
                  height: 300,
                  width: 300,
                ),
                SizedBox(
                  height: 20,
                ),
                Text(
                  caption,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 20,
                    color: Colors.white,
                  ),
                ),
                SizedBox(
                  height: 70,
                ),
                GestureDetector(
                  onTap: () {
                    getCaption();
                  },
                  child: Container(
                    height: 40,
                    width: 300,
                    color: Colors.white,
                    child: Center(
                      child: Text('Get Caption'),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Container(
                width: MediaQuery.of(context).size.width,
                padding: EdgeInsets.only(left: 20),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      width: 20,
                    ),
                    PageLink(
                      links: [
                        PageLinkInfo(
                          ease: Curves.easeOut,
                          duration: 0.2,
                          pageBuilder: () => HomePage(),
                        ),
                      ],
                      child: Container(
                        height: 60,
                        width: 60,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.all(
                            Radius.elliptical(9999.0, 9999.0),
                          ),
                          color: const Color(0xFFFFFFFF),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0x29000000),
                              offset: Offset(6, 6),
                              blurRadius: 6,
                            ),
                          ],
                        ),
                        child: Icon(
                          Icons.arrow_forward_rounded,
                          color: Color(0xBF0B1F51),
                        ),
                      ),
                    ),
                    SizedBox(
                      height: 100,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
