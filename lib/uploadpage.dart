import 'package:adobe_xd/adobe_xd.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'final_page.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'global.dart' as global;
import 'package:rflutter_alert/rflutter_alert.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_api.dart';

class UploadPage extends StatefulWidget {
  @override
  _UploadPageState createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  UploadTask task;
  File _image;
  final FirebaseFirestore _firebaseFirestore = FirebaseFirestore.instance;
  final ImagePicker _picker = ImagePicker();

  Future getImage() async {
    final img = await _picker.getImage(source: ImageSource.gallery);

    print(img);
    Alert(
            context: context,
            title: "Image Uploaded",
            desc:
                "Press on the arrow button to receive the caption for the image.")
        .show();

    setState(() {
      _image = File(img.path);
      global.image = img;
    });
  }

  Future uploadFile() async{
    if (_image == null) return;
    final destination = 'files/imgUpload';

    task =FirebaseApi.uploadFile(destination, _image);

    if (task == null) return;

    final snapshot = await task.whenComplete(() {});
    final urlDownload = await snapshot.ref.getDownloadURL();

    _firebaseFirestore.collection('Image').doc('uploadImage').set({
      'imgLink' : urlDownload
    });

    print('Download link = $urlDownload');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFDBE9F6),
      body: Stack(
        children: [
          new Image.asset("assets/images/uploadpage.jpg"),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  SizedBox(
                    width: 20,
                  ),
                  GestureDetector(
                    onTap: (){
                      uploadFile();
                      print('Hello');
                      Navigator.push(context, MaterialPageRoute(builder: (context)=> FinalPage(image: _image,)));
                    },
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
              Column(
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  SizedBox(
                    height: 200,
                  ),
                  GestureDetector(
                    onTap: getImage,
                    child: Container(
                      height: 60,
                      width: 150,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(10.0),
                        color: const Color(0xFFFFFFFF),
                      ),
                      child: Text(
                        '\nUPLOAD',
                        style: TextStyle(
                          fontFamily: 'Lucida Sans',
                          fontSize: 20,
                          color: const Color(0xBF0B1F51),
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ],
              ),
              Column(
                children: [
                  SizedBox(
                    width: 60,
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
