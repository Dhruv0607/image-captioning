import 'package:flutter/material.dart';
import 'package:image_caption/uploadpage.dart';
import 'home_page.dart';

void main() {
  runApp(ImageCaptioning());
}

class ImageCaptioning extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),
    );
  }
}
