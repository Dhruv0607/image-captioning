import 'package:adobe_xd/page_link.dart';
import 'package:flutter/material.dart';
import 'package:image_caption/home_page.dart';

class FinalPage extends StatefulWidget {
  @override
  _FinalPageState createState() => _FinalPageState();
}

class _FinalPageState extends State<FinalPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFDBE9F6),
      body: Stack(
        children: [
          new Image.asset("assets/images/finalpage.jpg"),
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
            ],
          ),
        ],
      ),
    );
  }
}
