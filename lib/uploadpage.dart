import 'package:adobe_xd/adobe_xd.dart';
import 'package:flutter/material.dart';
import 'final_page.dart';
import 'package:image_picker/image_picker.dart';
import 'global.dart' as global;
import 'package:rflutter_alert/rflutter_alert.dart';

class UploadPage extends StatefulWidget {
  @override
  _UploadPageState createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  final ImagePicker _picker = ImagePicker();

  Future getImage() async {
    final img = await _picker.getImage(source: ImageSource.gallery);

    Alert(
            context: context,
            title: "Image Uploaded",
            desc:
                "Press on the arrow button to receive the caption for the image.")
        .show();

    setState(() {
      global.image = img;
    });
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
                  PageLink(
                    links: [
                      PageLinkInfo(
                        ease: Curves.easeOut,
                        duration: 0.2,
                        pageBuilder: () => FinalPage(),
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
